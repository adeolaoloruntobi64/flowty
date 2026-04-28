use std::{cell::RefCell, rc::{Rc, Weak}};

use opencv::{
    core::*, imgproc, ximgproc
};

#[derive(Debug, Clone)]
pub struct Cluster {
    area: f64,
    index: usize,
    center: Point,
    is_bad: bool,
}

pub struct Cell {
    pub color: Vec3b,
    pub center: Point,
    pub is_dot: bool,
    pub is_recieving: bool,
    // Do note than a cell in one cannot be in the other
    pub neighbors: Vec<Weak<RefCell<Cell>>>,
    pub closeby: Vec<Weak<RefCell<Cell>>>,
    pub bbox: Rect,
    pub index: usize
}

impl Cell {
    pub fn new_def(
        color: Vec3b,
        center: Point,
        bbox: Rect,
        index: usize
    ) -> Self {
        Self {
            color,
            center,
            neighbors: Vec::new(),
            closeby: Vec::new(),
            is_dot: color != Vec3b::from_array([0, 0, 0]),
            is_recieving: false,
            bbox,
            index
        }
    }

    pub fn are_same(this: &Rc<RefCell<Cell>>, other: &Rc<RefCell<Cell>>) -> bool  {
        Rc::ptr_eq(this, other)
    }

    pub fn are_neighbors(this: &Rc<RefCell<Cell>>, neighbor: &Rc<RefCell<Cell>>) -> bool {
        this.borrow().neighbors.iter()
            .any(|c| {
                Weak::upgrade(&c).is_some_and(
                    |x| Rc::ptr_eq(&x, neighbor)
                )
            })
    }

    pub fn are_closeby(this: &Rc<RefCell<Cell>>, closeby: &Rc<RefCell<Cell>>) -> bool {
        this.borrow().closeby.iter()
            .any(|c| {
                Weak::upgrade(&c).is_some_and(
                    |x| Rc::ptr_eq(&x, closeby)
                )
            })
    }
    
    pub fn add_neighbor(this: &Rc<RefCell<Cell>>, neighbor: &Rc<RefCell<Cell>>) {
        if Self::are_same(this, neighbor)
        || Self::are_neighbors(this, neighbor) {
            return
        }
        this.borrow_mut().neighbors.push(Rc::downgrade(neighbor));
        neighbor.borrow_mut().neighbors.push(Rc::downgrade(this));
    }

    pub fn add_closeby(this: &Rc<RefCell<Cell>>, closeby: &Rc<RefCell<Cell>>) {
        if Self::are_same(this, closeby)
        || Self::are_closeby(this, closeby) {
            return
        }
        this.borrow_mut().closeby.push(Rc::downgrade(closeby));
        closeby.borrow_mut().closeby.push(Rc::downgrade(this));
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

#[derive(Debug, Default, Clone)]
pub struct DetectedMats {
    pub hsv_mat: Option<Mat>,
    pub edges_mat: Option<Mat>,
    pub colorthresh_mat: Option<Mat>,
    pub contour_mat: Option<Mat>,
    pub arrow_mat: Option<Mat>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DebugInfo {
    pub hsv_mat: bool,
    pub edges_mat: bool,
    pub colorthresh_mat: bool,
    pub contour_mat: bool,
    pub arrow_mat: bool,
    pub dilate_contours: bool
}

impl DebugInfo {
    #[allow(unused)]
    pub fn mostly_true() -> Self {
        Self {
            hsv_mat: true,
            edges_mat: true,
            colorthresh_mat: true,
            contour_mat: true,
            arrow_mat: true,
            dilate_contours: false
        }
    }
    #[allow(unused)]
    pub fn all_true() -> Self {
        Self {
            hsv_mat: true,
            edges_mat: true,
            colorthresh_mat: true,
            contour_mat: true,
            arrow_mat: true,
            dilate_contours: true
        }
    }
}

/*
Features:
- Edges
- Bridges
- Warps
- Windmills
- Alley
- Overpass
*/

// struct that holds reusable varables
#[derive(Debug, Default, Clone)]
pub struct CellDetector {
    hsv_mat: Mat,
    hold_mat: Mat,
    write_mat: Mat,
    temp_mat: Mat,
    thresh_mat: Mat,
    colorthresh_mat: Mat,
    and_mat: Mat,
    dilated_mat: Mat,
    dmats: DetectedMats,
}

impl CellDetector {

}



// Will change this to return a graph later
// On my computer, this takes ~0.1 seconds
// Later, not it takes ~0.2 seconds because of extra stuff
/*
I need to implement logic for
- Bridges (Template Matching + Remove)
- Windmill (Template Matching + Remove)
- Warps (Idk yet)
- Chain (In solver, assume chain, if not possible, then assume not chain. Or look at lvl name)
- Convert to PetGraph Grid

GOOD THING: Windmill and Bridges never rotate, so implementation should be
relatively easy. Chain is technically doable by assuming it is a chain in the
solver and if we can't find a solution, then we know it's not a chain

BAD THING: Warps...
What if I add edges, then be like, if have this edge, then it's a warp

For now, move on to the solver
Come back later to implement the others
*/
pub fn detect_cells(
    bit_mat: &Mat,
    debug_info: &DebugInfo
) -> opencv::error::Result<(Vec<Rc<RefCell<Cell>>>, DetectedMats)> {
    const DILATION_PIXELS: i32 = 4;
    const CLOSEBY_DILATION_PIXELS: i32 = 10;
    let mut hsv_mat = Mat::default();
    let mut write_mat = Mat::default();
    let mut temp_mat = Mat::default();
    let mut thresh_mat = Mat::default();
    let mut colorthresh_mat = Mat::default();
    let mut dmats = DetectedMats::default();
    // Convert to hsv (store in a temporary mat, we need to do some processing)
    imgproc::cvt_color(
        &bit_mat, 
        &mut temp_mat,
        imgproc::COLOR_BGR2HSV_FULL,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT
    )?;
    if debug_info.hsv_mat {
        dmats.hsv_mat = Some(temp_mat.clone());
    }
    // Remove small bg noise and unwanted segments
    extract_channel(
        &temp_mat,
        &mut write_mat,
        2
    )?;
    // used to be 35 with grayscale mat, but note ceil(15 * 255 / 100) = 39
    // The unwanted pixels have a v <= 15 on Paint
    // Later, lines are usually hav pretty high values, lowest I've seen
    // is like 25. ceil(25 * 255 / 100) = 64, So pick a number between 39
    // and 64 that works for all
    imgproc::threshold(
        &write_mat,
        &mut thresh_mat,
        40.,
        255.,
        imgproc::THRESH_BINARY
    )?;
    // hsv mat with all the noise gone
    temp_mat.copy_to_masked(&mut hsv_mat, &thresh_mat)?;
    // color mat with all the noise gone
    bit_mat.copy_to_masked(&mut colorthresh_mat, &thresh_mat)?;
    let mut arrow_mat = debug_info.arrow_mat.then(|| bit_mat.clone());
    // kernel to identify grid only (might identify parts of dot)
    let cirkernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(5, 5),
        Point::new(-1, -1)
    )?;
    if debug_info.edges_mat {
        dmats.edges_mat = Some(thresh_mat.clone());
    }
    // Isolate the circles
    imgproc::morphology_ex(
        &hsv_mat,
        &mut temp_mat,
        imgproc::MORPH_OPEN,
        &cirkernel,
        Point::new(-1, -1),
        6,
        BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    if debug_info.edges_mat {
        let mut m = Mat::default();
        imgproc::cvt_color(
            &temp_mat, 
            &mut m,
            imgproc::COLOR_HSV2BGR,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT
        )?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zc.png", &m).unwrap();
    }
    // dilate the circles
    imgproc::morphology_ex(
        &temp_mat,
        &mut write_mat,
        imgproc::MORPH_DILATE,
        &cirkernel,
        Point::new(-1, -1),
        8,
        BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    // Remove hold_mat
    absdiff(&hsv_mat, &write_mat, &mut temp_mat)?;
    if debug_info.edges_mat {
        let mut m = Mat::default();
        imgproc::cvt_color(
            &temp_mat, 
            &mut m,
            imgproc::COLOR_HSV2BGR,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT
        )?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zd.png", &m).unwrap();
    }
    // Get the stuff that clearly is not goog
    // 30 * 255 / 100 = 76.5; HSV is (0, 0, 0) - (180, 100, 100),
    // but opencv represents the max as (180, 255, 255); OR
    // (255, 255, 255) if using HSV_FULL
    in_range(
        &temp_mat,
        &Scalar::all(0.),
        &Scalar::new(3., 3., 100., 3.),
        &mut write_mat
    )?;
    temp_mat.set_scalar(0.into())?;
    // Make sure the pixels to dilate are colored
    write_mat.copy_to_masked(&mut temp_mat, &thresh_mat)?;
    // Usually, the very edges of the dot / ring aren't in the range, so we
    // dilate the area to remove just a tiny bit to make sure we get everything
    imgproc::dilate(
        &temp_mat,
        &mut write_mat,
        &cirkernel,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?
    )?;
    temp_mat.set_scalar(0.into())?;
    bitwise_not_def(&write_mat, &mut temp_mat)?;
    write_mat.set_scalar(0.into())?;
    hsv_mat.copy_to_masked(&mut write_mat, &temp_mat)?;
    extract_channel(
        &write_mat,
        &mut temp_mat,
        2
    )?;
    let extend_kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(13, 13),
        Point::new(-1, -1)
    )?;
    imgproc::dilate(
        &temp_mat,
        &mut write_mat,
        &extend_kernel,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    //if debug_info.edges_mat {
    //    dmats.edges_mat = Some(temp_mat.clone());
    //}
    imgproc::erode(
        &write_mat,
        &mut temp_mat,
        &extend_kernel,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    let mut contours = Vector::<Vector::<Point>>::new();
    // get the contours of the processed image
    imgproc::find_contours(
        &temp_mat,
        &mut contours,
        imgproc::RETR_LIST,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(-1, -1)
    )?;
    let mut contour_mat = if debug_info.contour_mat {
        Some(
            Mat::zeros_size(
                temp_mat.size()?,
                CV_8UC3,
            )?.to_mat()?
        )
    } else {
        None
    };
    let mut clusters = Vec::<Cluster>::with_capacity(contours.len());
    for (i, elem) in contours.iter().enumerate() {
        // (https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html#:~:text=From%20this%20moments,M%5B%27m00%27%5D.)
        let moments = imgproc::moments(&elem, true)?;
        let area = moments.m00;
        if area == 0. { continue }
        let center = Point::new(
            (moments.m10 / moments.m00) as i32,
            (moments.m01 / moments.m00) as i32,
        );
        let cluster = Cluster {
            area,
            index: i,
            center,
            is_bad: area < 600.0 || area > 60000.0
        };
        clusters.push(cluster);
    }    
    let mut cells: Vec<Rc<RefCell<Cell>>> = Vec::new();
    let imgbound = Rect::new(
        0,
        0,
        temp_mat.cols(),
        temp_mat.rows()
    );
    for i in 0..clusters.len() {
        // Make sure contour is not too big or too small
        if clusters[i].is_bad {
            if let Some(contour_mat) = &mut contour_mat {
                imgproc::draw_contours(
                    contour_mat,
                    &mut contours, // WHY IS THIS MUTABLE???
                    clusters[i].index as i32,
                    Scalar::all(255.),
                    1,
                    imgproc::LINE_8,
                    &no_array(),
                    i32::MAX,
                    Point::default(),
                )?;
            };
            continue
        }
        let color = color_from_h(
            (i * 12) as f64 * 360. / clusters.len() as f64
        );
        let contour = contours.get(clusters[i].index)?;
        if 'a: {
            for cell2 in &cells {
                let c2 = cell2.borrow();
                // A dot inside a cell will have approx the same center as the cell
                // Check if 2 centers are approx the same. If they are,
                // then delete the most recent one, which should be the circle
                // since we sort contours by size descending
                if distance_between(&clusters[i].center, &c2.center) < 12. {
                    break 'a true;
                }
            }
            false
        } {
            if let Some(contour_mat) = &mut contour_mat {
                imgproc::draw_contours(
                    contour_mat,
                    &mut contours,
                    clusters[i].index as i32,
                    color,
                    3,
                    imgproc::LINE_8,
                    &no_array(),
                    i32::MAX,
                    Point::default(),
                )?;
            }
            continue;
        }
        if let Some(contour_mat) = &mut contour_mat {
            imgproc::draw_contours(
                contour_mat,
                &mut contours,
                clusters[i].index as i32,
                color,
                1,
                imgproc::LINE_8,
                &no_array(),
                i32::MAX,
                Point::default(),
            )?;
        }
        let mut cell_color = colorthresh_mat.at_pt::<Vec3b>(clusters[i].center)?.clone();
        if brightness_from_scalar(&cell_color) < 0.1 {
            cell_color = Vec3b::from_array([0, 0, 0]); // Black
            if let Some(arrow_mat) = &mut arrow_mat {
                imgproc::circle(
                    arrow_mat,
                    clusters[i].center,
                    7,
                    Scalar::new(255., 215., 125., 255.),
                    1,
                    imgproc::LINE_8,
                    0
                )?;
            }
        } else {
            // Paranoia
            cell_color = Vec3b::from_array([
                cell_color[0] & 0b11111100,
                cell_color[1] & 0b11111100,
                cell_color[2] & 0b11111100,
            ]);
        }
        // Expanding the bounding box for ROI: (semilogs)
        let mut bbox = imgproc::bounding_rect(&contour)?;
        bbox.x -= 5 * DILATION_PIXELS;
        bbox.y -= 5 * DILATION_PIXELS;
        bbox.width += 10 * DILATION_PIXELS;
        bbox.height += 10 * DILATION_PIXELS;
        bbox &= imgbound;
        cells.push(Rc::new(RefCell::new(Cell::new_def(
            cell_color,
            clusters[i].center,
            bbox,
            clusters[i].index
        ))));
    }
    // Edge dilation adjacency check: (semilogs)
    let mut and_mat = Mat::default();
    let mut dilated_mat = Mat::default();
    let neighbor_kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(2 * DILATION_PIXELS + 1, 2 * DILATION_PIXELS + 1),
        Point::new(-1, -1)
    )?;
    let closeby_kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(2 * CLOSEBY_DILATION_PIXELS + 1, 2 * CLOSEBY_DILATION_PIXELS + 1),
        Point::new(-1, -1)
    )?;
    if debug_info.dilate_contours && let Some(contour_mat) = &mut contour_mat {
        let mut a = Mat::default();
        imgproc::dilate_def(contour_mat, &mut a, &closeby_kernel)?;
        *contour_mat = a;
    }
    for i in 0..cells.len() {
        for j in (i + 1)..cells.len() {
            if Cell::are_same(&cells[i], &cells[j])
            || Cell::are_neighbors(&cells[i], &cells[j])
            || Cell::are_closeby(&cells[i], &cells[j]) {
                continue
            }
            let roi = cells[i].borrow().bbox & cells[j].borrow().bbox;
            if roi.area() == 0 { continue }
            // This is fine because we set it to 0 later.
            unsafe {
                write_mat.create_size(roi.size(), CV_8U)?;
                dilated_mat.create_size(roi.size(), CV_8U)?;
                temp_mat.create_size(roi.size(), CV_8U)?
            };
            write_mat.set_scalar(0.into())?;
            temp_mat.set_scalar(0.into())?;
            dilated_mat.set_scalar(0.into())?;
            imgproc::draw_contours(
                &mut write_mat,
                &mut contours,
                cells[i].borrow().index as i32,
                Scalar::all(255.),
                1,
                imgproc::LINE_8,
                &no_array(),
                i32::MAX,
                roi.tl() * -1
            )?;
            imgproc::dilate_def(&write_mat, &mut dilated_mat, &neighbor_kernel)?;
            imgproc::draw_contours(
                &mut temp_mat,
                &mut contours,
                cells[j].borrow().index as i32,
                Scalar::all(255.),
                1,
                imgproc::LINE_8,
                &no_array(),
                i32::MAX,
                roi.tl() * -1
            )?;
            bitwise_and_def(&dilated_mat, &temp_mat, &mut and_mat)?;
            // Only count as neighbors if enough edge overlap exists:
            // count_non_zero(and_mat) >= 2*DILATION_PIXELS ensures real shared
            // edges pass while tiny corner/diagonal contacts with few
            // pixels are ignored
            if count_non_zero(&and_mat)? >= 2 * DILATION_PIXELS {
                if let Some(arrow_mat) = &mut arrow_mat {
                    imgproc::arrowed_line(
                        arrow_mat,
                        cells[i].borrow().center,
                        cells[j].borrow().center,
                        Scalar::all(255.),
                        1,
                        imgproc::LINE_8,
                        0,
                        0.1
                    )?;
                }
                Cell::add_neighbor(
                    &cells[i],
                &cells[j]
                );
                continue
            }
            and_mat.set_scalar(0.into())?;
            // Also want to see if it's closeby
            imgproc::dilate_def(&write_mat, &mut dilated_mat, &closeby_kernel)?;
            bitwise_and_def(&dilated_mat, &temp_mat, &mut and_mat)?;
            if count_non_zero(&and_mat)? >= 2 * CLOSEBY_DILATION_PIXELS {
                if let Some(arrow_mat) = &mut arrow_mat {
                    imgproc::arrowed_line(
                        arrow_mat,
                        cells[i].borrow().center,
                        cells[j].borrow().center,
                        Scalar::new(255., 0., 255., 255.),
                        1,
                        imgproc::LINE_8,
                        0,
                        0.1
                    )?;
                }
                Cell::add_closeby(
                    &cells[i],
                &cells[j]
                );
            }
        }
    }
    // Need to implement underpasses, mountain, windmill, and warps here as
    // (!c.is_dot && c.neighbors.len() == 1) below removes all empty 1 neighbor cells
    // Maybe if neighbor is 1 and next to edge of map, then check for mountain / warp
    // Actually, mountain, windmill, alley, and overpasses have recognizeable
    // configurations, so instead, just check for those
    cells.retain(|this| {
        // Think about edges, like [@. The @ will only
        // have 1 neighbor but it's fine as long as it's a color
        // If it isn't then something's wrong with the puzzle or
        // we scanned wrong
        // (!c.is_dot && c.neighbors.len() == 1) IS IMPORTANT
        // it is inactive for now as we need further processing
        !(
            (this.borrow().is_dot && this.borrow().neighbors.is_empty())
            // || (!this.borrow().is_dot && this.borrow().neighbors.len() == 1)
        )
    });
    if let Some(arrow_mat) = &mut arrow_mat {
        for cell in cells.iter()
            .map(|cell| cell.borrow()).filter(|cell| cell.is_dot) {
            imgproc::circle(
                arrow_mat,
                cell.center,
                7,
                Scalar::new(55., 15., 125., 255.),
                1,
                imgproc::FILLED,
                0
            )?;
        }
    }
    dmats.contour_mat = contour_mat;
    dmats.arrow_mat = arrow_mat;
    dmats.colorthresh_mat = debug_info.colorthresh_mat.then(
        || colorthresh_mat.clone()
    );
    return Ok((cells, dmats));
}