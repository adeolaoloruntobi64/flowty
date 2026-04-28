use std::{cell::RefCell, rc::Rc};

use opencv::{
    core::*, imgproc,
};

#[derive(Debug, Clone)]
pub struct Cluster {
    area: f64,
    index: usize,
    center: Point,
    is_bad: bool,
}

pub struct Line {
    start: Point,
    end: Point,
    index: usize,
    center: Point,
    length: f64,
}

pub struct Cell {
    pub color: Vec3b,
    pub center: Point,
    pub is_dot: bool,
    pub is_recieving: bool,
    pub neighbors: Vec<Rc<RefCell<Cell>>>
}

impl Cell {
    pub fn new_def(color: Vec3b, center: Point) -> Self {
        Self {
            color,
            center,
            neighbors: Vec::new(),
            is_dot: color != Vec3b::from_array([0, 0, 0]),
            is_recieving: false
        }
    }

    pub fn add_neighbor(this: &Rc<RefCell<Cell>>, neighbor: &Rc<RefCell<Cell>>) {
        // C# logic: if neighbor == this
        if Rc::ptr_eq(this, neighbor) { return }
        
        if this.borrow().neighbors.iter()
            .any(|c| Rc::ptr_eq(c, neighbor)) {
            return
        }
        this.borrow_mut().neighbors.push(Rc::clone(neighbor));
        neighbor.borrow_mut().neighbors.push(Rc::clone(this));
    }
}

impl Line {
    pub fn new_def(start: Point, end: Point, index: usize) -> Self {
        let center = Point::new(
            (start.x + end.x) / 2,
            (start.y + end.y) / 2
        );
        Self { start, end, index, center, length: 0.0}
    }
}

fn color_from_h(hue: f64) -> Scalar {
    // Hue is given in degrees (0-360)
    // Saturation and Value are given as 0-1

    let hi = (hue / 60.).floor() as i32 % 6;
    let f = hue / 60. - (hue / 60.).floor();

    let v = 255.;
    let p = 0.;
    let q = 255. * (1. - f);
    let t = 255. * f;

    match hi {
        0 => Scalar::new(p, t, v, 255.0),
        1 => Scalar::new(p, v, q, 255.0),
        2 => Scalar::new(t, v, p, 255.0),
        3 => Scalar::new(v, q, p, 255.0),
        4 => Scalar::new(v, p, t, 255.0),
        _ => Scalar::new(q, p, v, 255.0),
    }
}

fn center_of_contour(contour: &Vector<Point>) -> Result<Point, opencv::Error> {
    let moments = imgproc::moments(contour, true)?;
    Ok(Point::new(
        (moments.m10 / (moments.m00 + 1e-5)) as i32,
        (moments.m01 / (moments.m00 + 1e-5)) as i32,
    ))
}

fn brightness_from_scalar(pixel: &Vec3b) -> f64 {
    // OpenCV Vec3b order is B, G, R
    let b = pixel[0] as f64 / 255.0;
    let g = pixel[1] as f64 / 255.0;
    let r = pixel[2] as f64 / 255.0;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);

    (max + min) / 2.0
}

fn distance_between(a: &Point, b: &Point) -> f64 {
    return (
        (a.x - b.x).pow(2) as f64 + 
        (a.y - b.y).pow(2) as f64
    ).sqrt()
}

fn distance_point_to_line(p: &Point, l: &Line) -> Result<f64, opencv::Error> {
    let dx = l.end.x - l.start.x;
    let dy = l.end.y - l.start.y;
    let num = (dy * (p.x - l.start.x) - dy * (p.y - l.start.y)).abs() as f64;
    let dem = (dx.pow(2) as f64 + dy.pow(2) as f64).sqrt();
    if dem == 0. { 
        Err(opencv::Error::new(0, "Divide By 0"))
    } else {
        Ok(num / dem)
    }
}

pub struct DetectedMats {
    pub bit_mat: Mat,
    pub gray_mat: Mat,
    pub contour_mat: Mat,
    pub ret_mat: Mat,
}

// Will change this to return a graph later
// On my computer, this takes ~0.1 seconds
pub fn detect_cells(
    bit_mat: Mat,
) -> opencv::error::Result<(Vec<Rc<RefCell<Cell>>>, DetectedMats)> {
    let mut gray_mat = Mat::default();
    let mut write_mat = Mat::default();
    let mut temp_mat = Mat::default();
    let mut cfilter_mat = Mat::default();
    imgproc::cvt_color(
        &bit_mat, 
        &mut gray_mat,
        imgproc::COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT
    )?;
    imgproc::threshold(
        &gray_mat,
        &mut write_mat,
        30.0,
        255.0,
        imgproc::THRESH_BINARY
    )?;
    bitwise_and(
        &gray_mat,
        &write_mat,
        &mut temp_mat,
        &no_array()
    )?;
    bitwise_and(
        &bit_mat,
        &bit_mat,
        &mut cfilter_mat,
        &temp_mat
    )?;
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(3, 21),
        Point::new(-1, -1)
    )?;
    imgproc::morphology_ex(
        &temp_mat,
        &mut gray_mat,
        imgproc::MORPH_TOPHAT,
        &kernel,
        Point::new(-1, -1),
        4,
        BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    let mut contours = Vector::<Vector::<Point>>::new();
    imgproc::find_contours(
        &gray_mat,
        &mut contours,
        imgproc::RETR_LIST,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(-1, -1)
    )?;
    let mut contour_mat: Mat = Mat::zeros_size(
        gray_mat.size()?,
        CV_8UC3,
    )?.to_mat()?;
    let mut clusters = Vec::<Cluster>::with_capacity(contours.len());
    for (i, elem) in contours.iter().enumerate() {
        let area = imgproc::contour_area_def(&elem)?;
        let cluster = Cluster {
            area,
            index: i,
            center: Point::default(),
            is_bad: area < 1000.0 || area > 60000.0
        };
        clusters.push(cluster);
    }

    // Sort by area so that cell comes before dot later
    clusters.sort_by(|a, b| b.area.total_cmp(&a.area));
    
    let mut lines: Vec<Line> = Vec::new();
    let mut cells: Vec<Rc<RefCell<Cell>>> = Vec::new();
    let mut indx: usize = 0;

    for i in 0..clusters.len() {
        // Make sure contour is not too big or too small
        if clusters[i].is_bad {
            // imgproc::draw_contours(
            //     &mut contour_mat,
            //     &mut contours,
            //     clusters[i].index as i32,
            //     Scalar::all(255.),
            //     1,
            //     imgproc::LINE_8,
            //     &no_array(),
            //     i32::MAX,
            //     Point::default(),
            // )?;
            continue
        }
        let color = color_from_h(
            (i * 12) as f64 * 360. / clusters.len() as f64
        );
        let mut approx_contour = Vector::<Point>::new();
        let curve = &contours.get(clusters[i].index)?;
        let epsilon = 0.01 * imgproc::arc_length(
            &curve,
            true
        )?;
        imgproc::approx_poly_dp(
            &curve,
            &mut approx_contour,
            epsilon,
            true
        )?;
        // Make sure the contour actually has a chance to be a polygon
        // (Some contours are a line)
        if approx_contour.len() < 3 { 
            // imgproc::draw_contours(
            //     &mut contour_mat,
            //     &mut contours,
            //     clusters[i].index as i32,
            //     color,
            //     6,
            //     imgproc::LINE_8,
            //     &no_array(),
            //     i32::MAX,
            //     Point::default(),
            // )?;
            continue
        }
        // This is almost certainly a valid contour. proceed.
        let center: Point_<i32> = center_of_contour(&approx_contour)?;
        if 'a: {
            for cell2 in &cells {
                let c2 = cell2.borrow();
                // A dot inside a cell will have approx the same center as the cell
                // So, we need to check for duplicates. We know that a dot is always
                // smaller than a cell. So IF we wanted to, we could compare areas and
                // label the big one as the cell and the small one as the dot. However,
                // we are effectivelyt storing the data as Cell(is_dot: bool), so
                // instead, check if 2 centers are approx the same. If they are,
                // then delete the most recent one, which should be the circle
                if distance_between(&center, &c2.center) < 12. {
                    break 'a true;
                }
            }
            false
        } {
            // imgproc::draw_contours(
            //     &mut contour_mat,
            //     &mut contours,
            //     clusters[i].index as i32,
            //     color,
            //     3,
            //     imgproc::LINE_8,
            //     &no_array(),
            //     i32::MAX,
            //     Point::default(),
            // )?; 
            continue;
        }
        imgproc::draw_contours(
            &mut contour_mat,
            &mut contours,
            clusters[i].index as i32,
            color,
            imgproc::FILLED,
            imgproc::LINE_8,
            &no_array(),
            i32::MAX,
            Point::default(),
        )?; 
        for j in 0..(approx_contour.len() - 1) {
            lines.push(Line::new_def(
                approx_contour.get(j)?, 
                approx_contour.get(j + 1)?,
                indx
            ));
        }
        lines.push(Line::new_def(
            approx_contour.get(0)?, 
            approx_contour.get(approx_contour.len() - 1)?,
            indx
        ));
        let mut cell_color = cfilter_mat.at_pt::<Vec3b>(center)?.clone();
        if brightness_from_scalar(&cell_color) < 0.1 {
            cell_color = Vec3b::from_array([0, 0, 0]); // Black
            // imgproc::circle(
            //     &mut cfilter_mat,
            //     center,
            //     7,
            //     Scalar::new(155., 115., 125., 255.),
            //     1,
            //     imgproc::LINE_8,
            //     0
            // )?;
        } else {
            cell_color = Vec3b::from_array([
                cell_color[0] & 0b11111100,
                cell_color[1] & 0b11111100,
                cell_color[2] & 0b11111100,
            ]);
        }
        cells.push(Rc::new(RefCell::new(Cell::new_def(
            cell_color,
            center,
        ))));
        indx += 1;
    }

    let mut dilated = Mat::default();
    let kernel2 = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(1, 1),
        Point::new(-1, -1)
    )?;
    println!("{:?}", kernel2.size()?);
    let mut lines_mat: Mat = Mat::zeros_size(
        gray_mat.size()?,
        CV_8UC3,
    )?.to_mat()?;
    for (i, line) in lines.iter().enumerate() {
        let color = color_from_h(
            (i * 36)  as f64 * 360. / clusters.len() as f64
        );
        imgproc::line(
            &mut lines_mat,
            line.start,
            line.end, color, 1, imgproc::LINE_8, 0)?;
    }
    imgproc::dilate(
        &contour_mat,
        &mut dilated,
        &kernel,
        Point::new(-1, -1),
        2,
        BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    // 4-neighborhood offsets (NO diagonals)
    let dx = [-1, 1, 0, 0];
    let dy = [0, 0, 1, -1];

    for y in 0..contour_mat.rows() {
        for x in 0..contour_mat.cols() {

        }
    }

    //imgproc::connected_components(image, labels, connectivity, ltype)

    // I just copied this tbh
    // for i in 0..lines.len() {
    //     // If the line is too short, then it probably isn't an edge
    //     if distance_between(&lines[i].start, &lines[i].end) < 12. { continue }
    //     // For every other line
    //     for j in (i + 1)..lines.len() {
    //         // If these 2 lines come from the same contour or this 2nd line
    //         // is also too short, the don't look further
    //         if lines[i].index == lines[j].index 
    //         || distance_between(&lines[j].start, &lines[j].end) < 12. { 
    //             continue
    //         }
    //         // If the distance between 2 lines center's is small and each center
    //         // is within 5 pixels of the any point on the line
    //         if distance_between(&lines[i].center, &lines[j].center) < 12.
    //         && (
    //             distance_point_to_line(&lines[i].center, &lines[j])? < 15.0
    //             || distance_point_to_line(&lines[j].center, &lines[i])? < 15.0
    //         ) {
    //             if lines[i].index == 158 {
    //              imgproc::circle(
    //                 &mut cfilter_mat,
    //                 cells[lines[i].index].borrow().center,
    //                 7,
    //                 Scalar::new(0., 15., 125., 255.),
    //                 1,
    //                 imgproc::LINE_8,
    //                 0
    //             )?;
    //              imgproc::circle(
    //                 &mut cfilter_mat,
    //                 cells[lines[j].index].borrow().center,
    //                 7,
    //                 Scalar::new(55., 0., 125., 255.),
    //                 1,
    //                 imgproc::LINE_8,
    //                 0
    //             )?;}
    //             Cell::add_neighbor(
    //                 &cells[lines[i].index],
    //             &cells[lines[j].index]
    //             );
    //             println!("INDEX {i} is neighbors with INDEX {j}");
    //             println!("{} CONTOUR {}", lines[i].index, lines[j].index);
    //             println!("{:?} CELL {:?}", cells[lines[i].index].borrow().center, cells[lines[j].index].borrow().center);
    //         }
    //     }
    // }
    println!("{}", indx);
    let mut removables = Vec::new();
    for cell in &cells {
        let c = cell.borrow();
        // Think about edges, like [@. The @ will only
        // have 1 neighbor but it's fine as long as it's a color
        // If it isn't then something's wrong with the puzzle or
        // we scanned wrong
        //(isCellDot && Neighbors.Count == 0) || Neighbors.Count == 1;
        if (c.is_dot && c.neighbors.is_empty()) || (!c.is_dot && c.neighbors.len() == 1) {
            removables.push(cell.clone());
        }
    }

    cells.retain(|this| {
        !removables.iter().any(|other| {
            Rc::ptr_eq(this, other)
        })
    });

    let mut ret_mat = lines_mat;
    for cell in cells.iter()
        .map(|cell| cell.borrow()).filter(|cell| cell.is_dot) {
        //imgproc::circle(
        //    &mut ret_mat,
        //    cell.center,
        //    7,
        //    Scalar::new(55., 15., 125., 255.),
        //    1,
        //    imgproc::LINE_8,
        //    0
        //)?;
    }
    println!("{:?}", contours.get(158).unwrap());
    // I think I will have a map of points (cell centers) to node indices
    let dmat = DetectedMats { bit_mat, gray_mat, contour_mat, ret_mat };
    return Ok((cells, dmat));
}

let lmat = Mat::new_rows_cols_with_bytes::<u8>(
    temp_mat.rows(),
    temp_mat.cols(),
    &lightness
)?;
imgproc::threshold(
    &lmat,
    blob_mat,
    40.,
    255.,
    imgproc::THRESH_BINARY_INV
)?;
// Remove small bg noise and unwanted segments
// used to be 35 with grayscale mat, but note ceil(15 * 255 / 100) = 39
// Two obervations
// 1. The unwanted pixels and background lines have a v <= 15 on Paint
// 2. The blobs have a s >= 96
extract_channel(
    temp_mat,
    thresh_mat,
    2
)?;
// #2: Find pixels with low V values (store in write)
imgproc::threshold(
    thresh_mat,
    write_mat,
    40.,
    255.,
    imgproc::THRESH_BINARY_INV
)?;
// store the union of both in thresh
bitwise_or_def(write_mat, blob_mat, thresh_mat)?;
// #3: Get blobs
in_range(
    temp_mat,
    &Vec3b::from([0, 225, 0]),
    &Vec3b::from([255, 255, 90]),
    blob_mat
)?;
// store the union of all 3 in write
bitwise_or_def(thresh_mat, blob_mat, write_mat)?;
// Remember we want to get rid of what passed (white), so invert
bitwise_not_def(write_mat, thresh_mat)?;
// hsv mat with all the noise gone
temp_mat.copy_to_masked(hsv_mat, thresh_mat)?;
// color mat with all the noise gone
bit_mat.copy_to_masked(colorthresh_mat, thresh_mat)?;
let mut arrow_mat = debug_info.arrow_mat.then(|| bit_mat.clone());
if debug_info.edges_mat {
    dmats.edges_mat = Some(thresh_mat.clone());
}
// Isolate the circles
imgproc::morphology_ex(
    hsv_mat,
    temp_mat,
    imgproc::MORPH_OPEN,
    circle_kernel,
    Point::new(-1, -1),
    3,
    BORDER_CONSTANT,
    imgproc::morphology_default_border_value()?,
)?;
// dilate the circles (3 first, remove blobs, then 4)
imgproc::morphology_ex(
    temp_mat,
    write_mat,
    imgproc::MORPH_DILATE,
    circle_kernel,
    Point::new(-1, -1),
    3,
    BORDER_CONSTANT,
    imgproc::morphology_default_border_value()?,
)?;
// remove not (high saturation and high volume)
// Extract channel and thresh
in_range(
    write_mat,
    &Vec3b::from_array([0, 0, 100]),
    &Vec3b::from_array([255, 255, 255]),
    temp_mat
)?;
// stuff that is in temp, but not in write, must be removed from the original
bitwise_not_def(temp_mat, blob_mat)?;
write_mat.copy_to_masked(temp_mat, blob_mat)?;
imgproc::threshold(
    temp_mat,
    blob_mat,
    1.,
    255.,
    imgproc::THRESH_BINARY_INV
)?;
//other_mat.set_scalar(0.into())?;
hsv_mat.copy_to_masked(other_mat, blob_mat)?;
std::mem::swap(hsv_mat, other_mat);
temp_mat.set_scalar(0.into())?;
write_mat.copy_to_masked(temp_mat, blob_mat)?;
// The last 4 dilations
imgproc::morphology_ex(
    temp_mat,
    write_mat,
    imgproc::MORPH_DILATE,
    circle_kernel,
    Point::new(-1, -1),
    4,
    BORDER_CONSTANT,
    imgproc::morphology_default_border_value()?,
)?;
// Remove write_mat
absdiff(hsv_mat, write_mat, temp_mat)?;
// Alright so in the hsv space, hue is circular, i.e. 255 and 0 are almost
// the same hue. So imagine if the hue in the hsv_mat is 255, but it's 0
// in the write_mat. Very bad. So we need to sort of encode the cyclic
// nature of hsv
extract_channel(temp_mat, blob_mat, 0)?;
subtract_def(
    &Scalar::all(HUE_END),
    blob_mat,
    write_mat,
)?;
min(blob_mat, write_mat, and_mat)?;
insert_channel(and_mat, temp_mat, 0)?;

// Get the stuff that clearly is not good
in_range(
    temp_mat,
    &Vec3b::all(0),
    &Vec3b::from([5, 50, 80]),
    write_mat
)?;
temp_mat.set_scalar(0.into())?;
// Make sure the pixels to dilate are colored
write_mat.copy_to_masked(temp_mat, thresh_mat)?;
// Usually, the very edges of the dot / ring aren't in the range, so we
// dilate the area to remove just a tiny bit to make sure we get everything
imgproc::dilate(
    temp_mat,
    write_mat,
    circle_kernel,
    Point::new(-1, -1),
    2,
    BORDER_CONSTANT,
    imgproc::morphology_default_border_value()?
)?;
// Invert it so that we know where to not copy
bitwise_not_def(write_mat, temp_mat)?;
// We no longer need the full hsv mat. We only need the values now.
extract_channel(
    hsv_mat,
    write_mat,
    2
)?;
// Only copy pixels that aren't part of dots or rings
hsv_mat.set_scalar(0.into())?;
write_mat.copy_to_masked(hsv_mat, temp_mat)?;
// Some small segments of lines were erased during either the initial
// threshold, or from the removal of dots and rings. Of the cases where
// the lines are erased, the former makes up the majority. Dilate and erode
// the image to bridge those small gaps
// close(x) = erode(dilate(x))
imgproc::morphology_ex(
    hsv_mat,
    temp_mat,
    imgproc::MORPH_CLOSE,
    extend_kernel,
    Point::new(-1, -1),
    4,
    BORDER_CONSTANT,
    imgproc::morphology_default_border_value()?,
)?;

Computer Science: Comprehensive Stream
Computer Science: Software Engineering Stream
Computer Science: Entrepreneurship Stream
Computer Science: Information Systems Stream

Statistics: Quantitative Finance Stream
Statistics: Statistical Machine Learning and Data Science Stream
Statistics: Statistical Science Stream