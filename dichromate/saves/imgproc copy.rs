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

pub struct Cell {
    pub color: Vec3b,
    pub center: Point,
    pub is_dot: bool,
    pub is_recieving: bool,
    // Do note than a cell in one cannot be in the other
    pub neighbors: Vec<Rc<RefCell<Cell>>>,
    pub closeby: Vec<Rc<RefCell<Cell>>>,
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

    pub fn are_neighbors(this: &Rc<RefCell<Cell>>, neighbor: &Rc<RefCell<Cell>>) -> bool {
        Rc::ptr_eq(this, neighbor) || this.borrow().neighbors.iter()
            .any(|c| Rc::ptr_eq(c, neighbor))
    }

    pub fn are_closeby(this: &Rc<RefCell<Cell>>, closeby: &Rc<RefCell<Cell>>) -> bool {
        Rc::ptr_eq(this, closeby) || this.borrow().closeby.iter()
            .any(|c| Rc::ptr_eq(c, closeby))
    }
    
    pub fn add_neighbor(this: &Rc<RefCell<Cell>>, neighbor: &Rc<RefCell<Cell>>) {
        if Self::are_neighbors(this, neighbor) { return }
        this.borrow_mut().neighbors.push(Rc::clone(neighbor));
        neighbor.borrow_mut().neighbors.push(Rc::clone(this));
    }

    pub fn add_closeby(this: &Rc<RefCell<Cell>>, closeby: &Rc<RefCell<Cell>>) {
        if Self::are_closeby(this, closeby) { return }
        this.borrow_mut().closeby.push(Rc::clone(closeby));
        closeby.borrow_mut().closeby.push(Rc::clone(this));
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

#[derive(Debug, Default, Clone)]
pub struct DetectedMats {
    pub gray_mat: Option<Mat>,
    pub edges_mat: Option<Mat>,
    pub colorthresh_mat: Option<Mat>,
    pub contour_mat: Option<Mat>,
    pub arrow_mat: Option<Mat>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DebugInfo {
    pub gray_mat: bool,
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
            gray_mat: true,
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
            gray_mat: true,
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


// Will change this to return a graph later
// On my computer, this takes ~0.1 seconds
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
    let mut gray_mat = Mat::default();
    let mut write_mat = Mat::default();
    let mut temp_mat = Mat::default();
    let mut colorthresh_mat = Mat::default();
    let mut dmats = DetectedMats::default();
    // Convert to grayscale
    imgproc::cvt_color(
        &bit_mat, 
        &mut gray_mat,
        imgproc::COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT
    )?;
    if debug_info.gray_mat {
        dmats.gray_mat = Some(gray_mat.clone());
    }
    // Remove small bg noise and unwanted segments
    imgproc::adaptive_threshold(
        &gray_mat,
        &mut write_mat,
        255.,
        imgproc::ADAPTIVE_THRESH_MEAN_C,
        imgproc::THRESH_BINARY_INV,
        3,
        2.
    )?;
    // gray mat with all the noise gone
    //bitwise_and(
    //    &gray_mat,
    //    &write_mat,
    //    &mut temp_mat,
    //    &no_array()
    //)?;
    // color mat with all the noise gone
    //bitwise_and(
    //    &bit_mat,
    //    &bit_mat,
    //    &mut colorthresh_mat,
    //    &write_mat
    //)?;
    colorthresh_mat = bit_mat.clone();
    let mut arrow_mat = if debug_info.arrow_mat {
        Some(colorthresh_mat.clone())
    } else {
        None
    };
    //let kernel = imgproc::get_structuring_element(
    //    imgproc::MORPH_RECT,
    //    Size::new(3, 3),
    //    Point::new(-1, -1)
    //)?;
    // 4 iterations should be okay if I feature match and remove windmills
    // OR IF BLACK AND HAS 1 NEIGHBOR, CHECK ACROSS FOR BLACK AND ONE NEIGHBOR
    // IF FIND 3 PAIRS WITHIN RADIUS X, DECLARE IT A WINDMILL
    // CAN PROB DO SOMETHING SIMILAR FOR BRIDGES TOO
    // actually, for windmill, look for 2 pairs with no neighbors and one pair
    // with 2 long distance neighbors
    //imgproc::morphology_ex(
    //    &temp_mat,
    //    &mut gray_mat,
    //    imgproc::MORPH_TOPHAT,
    //    &kernel,
    //    Point::new(-1, -1),
    //    5,
    //    BORDER_CONSTANT,
    //    imgproc::morphology_default_border_value()?,
    //)?;
    write_mat.convert_to_def(&mut gray_mat, CV_8UC3)?;
    write_mat = gray_mat.clone();
    //std::mem::swap(&mut gray_mat, &mut write_mat);
    if debug_info.edges_mat {
        dmats.edges_mat = Some(gray_mat.clone());
    }
    let mut contours = Vector::<Vector::<Point>>::new();
    imgproc::find_contours(
        &gray_mat,
        &mut contours,
        imgproc::RETR_LIST,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(-1, -1)
    )?;
    let mut contour_mat = if debug_info.contour_mat {
        Some(Mat::zeros_size(
            gray_mat.size()?,
            CV_8UC3,
        )?.to_mat()?)
    } else {
        None
    };
    let mut clusters = Vec::<Cluster>::with_capacity(contours.len());
    for (i, elem) in contours.iter().enumerate() {
        let area = imgproc::contour_area_def(&elem)?;
        let cluster = Cluster {
            area,
            index: i,
            center: Point::default(),
            is_bad: area < 600.0 || area > 60000.0
        };
        clusters.push(cluster);
    }

    // Sort by descending area so that cell comes before dot
    // Later in development, we don't really care about identifying the dot
    // But keep this just in case we do detect it
    clusters.sort_by(|a, b| b.area.total_cmp(&a.area));
    
    let mut cells: Vec<Rc<RefCell<Cell>>> = Vec::new();
    let imgbound = Rect::new(
        0,
        0,
        gray_mat.cols(),
        gray_mat.rows()
    );
    for i in 0..clusters.len() {
        // Make sure contour is not too big or too small
        let color = color_from_h(
            (i * 12) as f64 * 360. / clusters.len() as f64
        );
        if clusters[i].is_bad {
            if let Some(contour_mat) = &mut contour_mat {
                imgproc::draw_contours(
                    contour_mat,
                    &mut contours, // WHY IS THIS MUTABLE???
                    clusters[i].index as i32,
                    color,
                    1,
                    imgproc::LINE_8,
                    &no_array(),
                    i32::MAX,
                    Point::default(),
                )?;
            };
            continue
        }
        
        let contour = contours.get(clusters[i].index)?;
        let center: Point_<i32> = center_of_contour(&contour)?;
        if 'a: {
            for cell2 in &cells {
                let c2 = cell2.borrow();
                // A dot inside a cell will have approx the same center as the cell
                // Check if 2 centers are approx the same. If they are,
                // then delete the most recent one, which should be the circle
                // since we sort contours by size descending
                if distance_between(&center, &c2.center) < 12. {
                    break 'a true;
                }
            }
            false
        } {
            println!("CENTRES");
            if let Some(contour_mat) = &mut contour_mat {
                //imgproc::draw_contours(
                //    contour_mat,
                //    &mut contours,
                //    clusters[i].index as i32,
                //    Scalar::all(255.),
                //    3,
                //    imgproc::LINE_8,
                //    &no_array(),
                //    i32::MAX,
                //    Point::default(),
                //)?;
            }
            continue;
        }
        if let Some(contour_mat) = &mut contour_mat {
            println!("YUH");
            //imgproc::draw_contours(
            //    contour_mat,
            //    &mut contours,
            //    clusters[i].index as i32,
            //    color,
            //    1,
            //    imgproc::LINE_8,
            //    &no_array(),
            //    i32::MAX,
            //    Point::default(),
            //)?;
        }
        let mut cell_color = colorthresh_mat.at_pt::<Vec3b>(center)?.clone();
        if brightness_from_scalar(&cell_color) < 0.1 {
            cell_color = Vec3b::from_array([0, 0, 0]); // Black
            if let Some(arrow_mat) = &mut arrow_mat {
                imgproc::circle(
                    arrow_mat,
                    center,
                    7,
                    Scalar::new(255., 215., 125., 255.),
                    1,
                    imgproc::LINE_8,
                    0
                )?;
            }
        } else {
            cell_color = Vec3b::from_array([
                cell_color[0] & 0b11111100,
                cell_color[1] & 0b11111100,
                cell_color[2] & 0b11111100,
            ]);
        }
        /*
        Expanding the bounding box for ROI:
        1. We take the bounding box of a cell (rect) and expand it by DILATION_PIXELS
        in all directions.
        - When we dilate the cell's edges later, we don't get clipping at the
            edges of the ROI.
        2. Then we intersect with `imgbound` (the full image bounds) to ensure
        the expanded rect **stays inside the image**.
        -This leaves us with a slightly larger ROI that fully contains the cell
            and any dilated edges.
        */
        let mut bbox = imgproc::bounding_rect(&contour)?;
        bbox.x -= 5 * DILATION_PIXELS;
        bbox.y -= 5 * DILATION_PIXELS;
        bbox.width += 10 * DILATION_PIXELS;
        bbox.height += 10 * DILATION_PIXELS;
        bbox &= imgbound;
        cells.push(Rc::new(RefCell::new(Cell::new_def(
            cell_color,
            center,
            bbox,
            clusters[i].index
        ))));
    }

    /*
    Edge dilation adjacency check:
    1. Draw each cell's edges as a thin binary mask.
    2. Dilate edges by 'DILATION_PIXELS' pixels using an elliptical kernel.
    - Creates a small "buffer" around edges to bridge gaps or fragmented contours.
    - Elliptical kernel reduces the amount diagonal contact overlap.
    3. Draw candidate neighbor cell's edges as a thin binary mask.
    4. Compute bitwise AND between dilated edges and candidate edges.
    - The idea is we check how many pixels of the edges of cell b are in the
        dilation area of cell a. This also makes it so that corners don't
        qualify as a neighbor as there wouldn't be enough of those edge pixels
        in the dilation area
    5. If overlap >= minSharedPixels, then the 2 cells are neighbors.
     */
    // Since we're no longer using them, why not reuse them
    let mut edge1 = write_mat;
    let mut edge2 = temp_mat;
    let mut tmp = Mat::default();
    let mut dilated = Mat::default();
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
    for i in 0.. cells.len() {
        // let mut inspect = false;
        // if 100 <= cells[i].borrow().center.x && cells[i].borrow().center.x <= 140
        // && 600 <= cells[i].borrow().center.y && cells[i].borrow().center.y <= 640 
        // {
        //     println!("HERE");
        //     inspect = true;
        // }
        for j in (i + 1).. cells.len() {

            let mut pair = {
                (
                    200 <= cells[i].borrow().center.x && cells[i].borrow().center.x <= 240
                    && 440 <= cells[i].borrow().center.y && cells[i].borrow().center.y <= 480 
                    && 290 <= cells[j].borrow().center.x && cells[j].borrow().center.x <= 330
                    && 450 <= cells[j].borrow().center.y && cells[j].borrow().center.y <= 500
                ) || (
                    200 <= cells[j].borrow().center.x && cells[j].borrow().center.x <= 240
                    && 440 <= cells[j].borrow().center.y && cells[j].borrow().center.y <= 480 
                    && 290 <= cells[i].borrow().center.x && cells[i].borrow().center.x <= 330
                    && 450 <= cells[i].borrow().center.y && cells[i].borrow().center.y <= 500
                )
            };
            // MAYBE do this again but slightly bigger
            // Keep a list of almost neighbors
            // If len(almost) / len(neighbors + almost) > 0.6 (0.66 is 2/3 for triangle)
            // This means it is mostly surrounded by boundary edges
            // if ~ 2/3 check for mountain (0.61-0.70)
            // if ~3/4 check for underpass (0.70-0.79)
            // (60, 80) exclusive
            if Cell::are_neighbors(&cells[i], &cells[j]) { continue }
            let roi = cells[i].borrow().bbox & cells[j].borrow().bbox;
            if roi.area() == 0 { 
                if pair {
                    println!("R1: {:?}, R2: {:?}, AREA IS 0", cells[i].borrow().bbox , cells[j].borrow().bbox )
                }
                continue;
             }
            if pair {
                println!("NOT 0");
            }
            // Who knows why this is unsafe, cuz I don't
            unsafe {
                edge1.create_size(roi.size(), CV_8U)?;
                dilated.create_size(roi.size(), CV_8U)?;
                edge2.create_size(roi.size(), CV_8U)?
            };
            edge1.set_scalar(0.into())?;
            edge2.set_scalar(0.into())?;
            imgproc::draw_contours(
                &mut edge1,
                &mut contours,
                cells[i].borrow().index as i32,
                Scalar::all(255.),
                1,
                imgproc::LINE_8,
                &no_array(),
                i32::MAX,
                roi.tl() * -1
            )?;
            imgproc::dilate_def(&edge1, &mut dilated, &neighbor_kernel)?;
            imgproc::draw_contours(
                &mut edge2,
                &mut contours,
                cells[j].borrow().index as i32,
                Scalar::all(255.),
                1,
                imgproc::LINE_8,
                &no_array(),
                i32::MAX,
                roi.tl() * -1
            )?;
            bitwise_and_def(&dilated, &edge2, &mut tmp)?;
            // Only count as neighbors if enough edge overlap exists:
            // count_non_zero(tmp) >= 2*DILATION_PIXELS ensures real shared
            // edges pass while tiny corner/diagonal contacts with few
            // pixels are ignored
            if count_non_zero(&tmp)? >= 2 * DILATION_PIXELS {
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
            // Also want to see if it's closeby
            // Who knows why this is unsafe, cuz I don't
            unsafe {
                dilated.create_size(roi.size(), CV_8U)?;
            };
            imgproc::dilate_def(&edge1, &mut dilated, &closeby_kernel)?;
            bitwise_and_def(&dilated, &edge2, &mut tmp)?;
            if pair {
                use opencv::imgcodecs;
                pair = false;
                imgcodecs::imwrite_def("dichromate/pics2/AAA.png", &tmp).unwrap();
                println!("DONE");
            }
            if count_non_zero(&tmp)? >= 2 * CLOSEBY_DILATION_PIXELS {
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
    // Maybe if neighbor is 1 and next to edge of map, then check for mountain/ warp
    let mut removables = Vec::new();
    for cell in &cells {
        let c = cell.borrow();
        // Think about edges, like [@. The @ will only
        // have 1 neighbor but it's fine as long as it's a color
        // If it isn't then something's wrong with the puzzle or
        // we scanned wrong
        // (!c.is_dot && c.neighbors.len() == 1) IS IMPORTANT
        if (c.is_dot && c.neighbors.is_empty()) || (!c.is_dot && c.neighbors.len() == 1) {
            removables.push(cell.clone());
        }
    }
    cells.retain(|this| {
        !removables.iter().any(|other| {
            // THIS MIGHT NOT WORK AS SOME CELLS ARE NEIGHBORS
            // USE WEAK POINTERS
            Rc::ptr_eq(this, other)
        })
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