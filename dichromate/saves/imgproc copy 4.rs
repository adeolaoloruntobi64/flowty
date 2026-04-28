use std::{cell::RefCell, rc::Rc};
use opencv::{
    core::*, imgproc, ximgproc
};
use petgraph::prelude::UnGraphMap;
use rayon::prelude::*;

const DILATION_PIXELS: i32 = 4;
const CLOSEBY_DILATION_PIXELS: i32 = 10;
const HSV_TYPE: i32 = imgproc::COLOR_BGR2HSV_FULL;
const HUE_END: f64 = if HSV_TYPE == imgproc::COLOR_BGR2HSV_FULL {
    256.
} else if HSV_TYPE == imgproc::COLOR_BGR2HSV {
    180.
} else {
    panic!("Invalid HSV Type");
};

#[derive(Debug, Clone)]
pub struct Cluster {
    area: f64,
    index: usize,
    center: Point,
    is_bad: bool,
}

#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Location {
    pub x: i32,
    pub y: i32
}

impl From<Point> for Location {
    fn from(value: Point) -> Self {
        Self { x: value.x, y: value.y }
    }
}

impl Into<Point> for Location {
    fn into(self) -> Point {
        Point::new(self.x, self.y)
    }
}

impl Location {
    pub fn dne() -> Self {
        return Self { x: -1, y: -1 }
    }

    pub fn is_dne(&self) -> bool {
        return *self == Self::dne()
    }
}

#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GraphCell {
     pub affiliation: usize,
     pub center: Location,
     pub is_dot: bool,
     pub is_phantom: bool
}

impl GraphCell {
    pub fn new_def(
        affiliation: usize,
        center: Location,
        is_dot: bool,
        is_phantom: bool
    ) -> Self {
        Self { affiliation, center, is_dot, is_phantom }
    }
}

#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GraphEdge {
     pub affiliation: usize
}

pub struct Cell {
    pub color: Vec3b,
    pub center: Point,
    pub is_dot: bool,
    pub is_phantom: bool,
    // Do note than a cell in one cannot be in the other
    pub neighbors: Vec<Rc<RefCell<Cell>>>,
    pub closeby: Vec<Rc<RefCell<Cell>>>,
    pub bbox: Rect,
    pub index: usize,
    pub affiliation: usize
}

impl Cell {
    pub fn new_def(
        color: Vec3b,
        center: Point,
        bbox: Rect,
        index: usize,
        is_phantom: bool,
    ) -> Self {
        Self {
            color,
            center,
            neighbors: Vec::new(),
            closeby: Vec::new(),
            is_dot: color != Vec3b::from_array([0, 0, 0]),
            is_phantom,
            bbox,
            index,
            affiliation: 0
        }
    }

    pub fn are_same(this: &Rc<RefCell<Cell>>, other: &Rc<RefCell<Cell>>) -> bool  {
        Rc::ptr_eq(this, other)
    }

    pub fn are_neighbors(this: &Rc<RefCell<Cell>>, neighbor: &Rc<RefCell<Cell>>) -> bool {
        this.borrow().neighbors.iter()
            .any(|c| Rc::ptr_eq(&c, neighbor))
    }

    pub fn are_closeby(this: &Rc<RefCell<Cell>>, closeby: &Rc<RefCell<Cell>>) -> bool {
        this.borrow().closeby.iter()
            .any(|c| Rc::ptr_eq(&c, closeby))
    }
    
    pub fn add_neighbor(this: &Rc<RefCell<Cell>>, neighbor: &Rc<RefCell<Cell>>) {
        if Self::are_same(this, neighbor)
        || Self::are_neighbors(this, neighbor) {
            return
        }
        this.borrow_mut().neighbors.push(Rc::clone(neighbor));
        neighbor.borrow_mut().neighbors.push(Rc::clone(this));
    }

    pub fn add_closeby(this: &Rc<RefCell<Cell>>, closeby: &Rc<RefCell<Cell>>) {
        if Self::are_same(this, closeby)
        || Self::are_closeby(this, closeby) {
            return
        }
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

fn slope_between(a: &Point, b: &Point) -> f64 {
    (b.y - a.y) as f64 / (b.x - a.x) as f64
}

fn build_srgb_to_linear_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];

    for i in 0..256 {
        let c = i as f32 / 255.0;

        lut[i] = if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
    }

    lut
}

fn approx_equal_slopes(p1: Point, p2: Point, center: Point) -> bool {
    let dx1 = p1.x - center.x;
    let dy1 = p1.y - center.y;
    let dx2 = p2.x - center.x;
    let dy2 = p2.y - center.y;

    // Check if vectors are parallel: dx1*dy2 == dx2*dy1
    let cross = (dx1*dy2 - dx2*dy1) as f64;
    let max_mag = ((dx1.abs().max(dy1.abs())) * (dx2.abs().max(dy2.abs()))) as f64;
    cross.abs() <= 1e-12f64.max(1e-9f64 * max_mag)
}

fn build_lstar_lut() -> [u8; 4096] {
    let size = 4096; // resolution (tune this)
    let mut lut = [0; 4096];

    for i in 0..size {
        let y = i as f32 / (size as f32 - 1.0);

        let l = if y > 0.008856 { // y > EPSILON
            116.0 * y.cbrt() - 16.0
        } else { // KAPPA * Y
            903.3 * y
        };

        lut[i] = (l * 255.0 / 100.0) as u8;
    }

    lut
}

/// Compute L* (OpenCV-scaled 0–255) from BGR image buffer
/// input: &[u8] in BGRBGR... format
/// output: Vec<f32> same pixel count
pub fn fast_lstar(
    bgr: &[u8],
    gamma_lut: &[f32; 256],
    lstar_lut: &[u8; 4096],
    out: &mut Vec<u8>,
) {
    const CHUNK_SIZE: usize = 8192;
    let n_pixels = bgr.len() / 3;
    if out.capacity() < n_pixels {
        let additional = n_pixels - out.len();
        out.reserve(additional);
    }
    unsafe { out.set_len(bgr.len() / 3); }
    let out = out.as_mut_slice();
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            for (j, out_px) in out_chunk.iter_mut().enumerate() {
                let i = start + j;
                let idx = i * 3;

                let r_lin = gamma_lut[bgr[idx + 2] as usize];
                let g_lin = gamma_lut[bgr[idx + 1] as usize];
                let b_lin = gamma_lut[bgr[idx] as usize];

                let y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin;
                let lut_idx = (y * 4095.0) as usize;

                *out_px = lstar_lut[lut_idx];
            }
        });
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
    pub dilate_contours: bool,
    pub debug_save: bool,
    pub print_phases: bool,
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
            dilate_contours: false,
            debug_save: false,
            print_phases: false,
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
            dilate_contours: true,
            debug_save: true,
            print_phases: true,
        }
    }
    #[allow(unused)]
    pub fn all_false() -> Self {
        Self {
            hsv_mat: false,
            edges_mat: false,
            colorthresh_mat: false,
            contour_mat: false,
            arrow_mat: false,
            dilate_contours: false,
            debug_save: false,
            print_phases: false,
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
#[derive(Debug, Clone)]
pub struct CellDetector {
    hsv_mat: Mat,
    write_mat: Mat,
    temp_mat: Mat,
    blob_mat: Mat,
    thresh_mat: Mat,
    colorthresh_mat: Mat,
    hsvthresh_mat: Mat,
    and_mat: Mat,
    dilated_mat: Mat,
    circle_kernel: Mat,
    extend_kernel: Mat,
    neighbor_kernel: Mat,
    closeby_kernel: Mat,
    affiliation_displays: Vec<Vec3b>,
    lightness: Vec<u8>,
    gamma_lut: [f32; 256],
    lstar_lut: [u8; 4096]
}

impl CellDetector {
    pub fn new() -> opencv::error::Result<Self> {
        Ok(Self {
            hsv_mat: Mat::default(),
            write_mat: Mat::default(),
            temp_mat: Mat::default(),
            blob_mat: Mat::default(),
            thresh_mat: Mat::default(),
            colorthresh_mat: Mat::default(),
            hsvthresh_mat: Mat::default(),
            and_mat: Mat::default(),
            dilated_mat: Mat::default(),
            // kernel to identify grid only (might identify parts of dot)
            circle_kernel: imgproc::get_structuring_element(
                imgproc::MORPH_ELLIPSE,
                Size::new(5, 5),
                Point::new(-1, -1)
            )?,
            // kernel to dilate and erode the grid by to connect broken lines
            extend_kernel: imgproc::get_structuring_element(
                imgproc::MORPH_RECT,
                Size::new(7, 7),
                Point::new(-1, -1)
            )?,
            neighbor_kernel: imgproc::get_structuring_element(
                imgproc::MORPH_ELLIPSE,
                Size::new(2 * DILATION_PIXELS + 1, 2 * DILATION_PIXELS + 1),
                Point::new(-1, -1)
            )?,
            closeby_kernel: imgproc::get_structuring_element(
                imgproc::MORPH_ELLIPSE,
                Size::new(2 * CLOSEBY_DILATION_PIXELS + 1, 2 * CLOSEBY_DILATION_PIXELS + 1),
                Point::new(-1, -1)
            )?,
            // Our internal reference for terminal nodes
            affiliation_displays: Vec::new(),
            // Array to store CIE LAB Lightness of pixels
            lightness: Vec::new(),
            // Look up tables for conversion from bgr to L (AB are not found)
            gamma_lut: build_srgb_to_linear_lut(),
            lstar_lut: build_lstar_lut()
        })
    }

    fn clean(&mut self) -> opencv::Result<()> {
        self.hsv_mat.set_scalar(0.into())?;
        self.write_mat.set_scalar(0.into())?;
        self.temp_mat.set_scalar(0.into())?;
        self.blob_mat.set_scalar(0.into())?;
        self.thresh_mat.set_scalar(0.into())?;
        self.colorthresh_mat.set_scalar(0.into())?;
        self.hsvthresh_mat.set_scalar(0.into())?;
        self.and_mat.set_scalar(0.into())?;
        self.dilated_mat.set_scalar(0.into())?;
        self.affiliation_displays.clear();
        Ok(())
    }

    pub fn detect_cells(
        &mut self,
        bit_mat: &Mat,
        debug_info: &DebugInfo,
    ) -> opencv::error::Result<(UnGraphMap<GraphCell, GraphEdge>, DetectedMats)> {
        self.clean()?;
        detect_cells(
            &bit_mat,
            &debug_info,
            &self.circle_kernel,
            &self.extend_kernel,
            &self.neighbor_kernel,
            &self.closeby_kernel,
            &mut self.hsv_mat,
            &mut self.write_mat,
            &mut self.temp_mat,
            &mut self.blob_mat,
            &mut self.thresh_mat,
            &mut self.colorthresh_mat,
            &mut self.hsvthresh_mat,
            &mut self.and_mat,
            &mut self.dilated_mat,
            &mut self.affiliation_displays,
            &mut self.lightness,
            &self.gamma_lut,
            &self.lstar_lut
        )
    }
}



// Will change this to return a graph later
// On my computer, this takes ~0.1 seconds
// Later, not it takes ~0.2 seconds because of extra stuff
// Even later, it now takes 0.3 seconds
/*
I need to implement logic for
- Bridges (Template Matching + Remove) // Don't need template matching
- Windmill (Template Matching + Remove) // Don't need template matching
- Warps (Idk yet)
- Chain (In solver, assume chain, if not possible, then assume not chain. Or look at lvl name)
- Convert to PetGraph Grid

GOOD THING: Windmill and Bridges never rotate, so implementation should be
relatively easy. Chain is technically doable by assuming it is a chain in the
solver and if we can't find a solution, then we know it's not a chain

BAD THING: Warps...
What if I add edges, then be like, if have this edge, then it's a warp
This is pretty hard to do icl, especially with 

For now, move on to the solver
Come back later to implement the others
*/
// I originally defined this as a function before moving it into a struct
// Yes, this is long, and potentially inefficient. A detector specialized for
// rectangular grids will be way, WAY faster than this. But this function is
// meant to work for ALL flow free levels, so generalizability is the absolute
// priority, followed by performance
// Goal is 0.1s for small map
#[inline(always)]
fn detect_cells(
    bit_mat: &Mat,
    debug_info: &DebugInfo,
    circle_kernel: &Mat,
    extend_kernel: &Mat,
    neighbor_kernel: &Mat,
    closeby_kernel: &Mat,
    hsv_mat: &mut Mat,
    write_mat: &mut Mat,
    temp_mat: &mut Mat,
    blob_mat: &mut Mat,
    thresh_mat: &mut Mat,
    colorthresh_mat: &mut Mat,
    other_mat: &mut Mat,
    and_mat: &mut Mat,
    dilated_mat: &mut Mat,
    affiliation_displays: &mut Vec<Vec3b>,
    lightness: &mut Vec<u8>,
    gamma_lut: &[f32; 256],
    lstar_lut: &[u8; 4096]
) -> opencv::error::Result<(UnGraphMap<GraphCell, GraphEdge>, DetectedMats)> {
    let mut dmats = DetectedMats::default();
    // Convert to hsv (store in a temporary mat, we need to do some processing)
    imgproc::cvt_color(
        bit_mat, 
        temp_mat,
        HSV_TYPE,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT
    )?;
    if debug_info.hsv_mat {
        dmats.hsv_mat = Some(temp_mat.clone());
    }
    // What we really need is LSV, L from CIE LAB, and SV from HSV
    // Obviously, we have to do this ourselves, because "LSV" isn't an
    // actual color space, merely a convenient tool to employ
    let bgr = bit_mat.data_bytes()?;
    let t1 = std::time::Instant::now();
    // Currently takes 0.04s - 0.05s, Would prefer if it was 10x faster
    fast_lstar(bgr, gamma_lut, lstar_lut, lightness);
	let t2 = std::time::Instant::now();
    if debug_info.print_phases {
        println!("L* Took: {} seconds", (t2 - t1).as_secs_f32());
    }
    let lmat = Mat::new_rows_cols_with_bytes::<u8>(
        temp_mat.rows(),
        temp_mat.cols(),
        &lightness
    )?;
    // #1: Find pixels with low L values (store in blob)
    imgproc::threshold(
        &lmat,
        blob_mat,
        40.,
        255.,
        imgproc::THRESH_BINARY_INV
    )?;
    // #2: Find pixels with low V values (store in write)
    // Remove small bg noise and unwanted segments
    // used to be 35 with grayscale mat, but note ceil(15 * 255 / 100) = 39
    // Two obervations
    // 1. The unwanted pixels and background lines have a v <= 15 on Paint
    // 2. The blobs have a s >= 96
    in_range(
        temp_mat,
        &Vec3b::from_array([0, 0, 0]),
        &Vec3b::from_array([255, 255, 40]),
        write_mat,
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
    bitwise_and(temp_mat, temp_mat, hsv_mat, thresh_mat)?;
    // color mat with all the noise gone
    bitwise_and(bit_mat, bit_mat, colorthresh_mat, thresh_mat)?;
    if debug_info.debug_save {
        let mut m = Mat::default();
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zct.png", colorthresh_mat).unwrap();
    }
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
    //opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_hsv.png", hsv_mat).unwrap();
    if debug_info.debug_save {
        let mut m = Mat::default();
        //opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zctp.png", &x).unwrap();
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zct.png", temp_mat).unwrap();
    }
    if debug_info.debug_save {
        let mut m = Mat::default();
        imgproc::cvt_color(
            temp_mat, 
            &mut m,
            imgproc::COLOR_HSV2BGR,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT
        )?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zc.png", &m).unwrap();
    }
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
    bitwise_and(write_mat, write_mat, temp_mat, blob_mat)?;
    in_range(
        temp_mat,
        &Vec3b::from_array([0, 0, 0]),
        &Vec3b::from_array([1, 1, 1]),
        blob_mat,
    )?;
    bitwise_and(hsv_mat, hsv_mat, other_mat, blob_mat)?;
    std::mem::swap(hsv_mat, other_mat);
    bitwise_and(write_mat, write_mat, temp_mat, blob_mat)?;
    if debug_info.debug_save {
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zcp0.png", temp_mat).unwrap();
    }
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
    if debug_info.debug_save {
        let mut m = Mat::default();
        imgproc::cvt_color(
            write_mat, 
            &mut m,
            imgproc::COLOR_HSV2BGR,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT
        )?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zd.png", &m).unwrap();
    }
    // Remove write_mat
    absdiff(hsv_mat, write_mat, temp_mat)?;
    if debug_info.debug_save {
        let mut m = Mat::default();
        let mut m2 = Mat::default();
        imgproc::cvt_color(
            temp_mat, 
            &mut m,
            imgproc::COLOR_HSV2BGR_FULL,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT
        )?;
        imgproc::cvt_color(
            hsv_mat, 
            &mut m2,
            imgproc::COLOR_HSV2BGR_FULL,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT
        )?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zd2.png", &m).unwrap();
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zd2b.png", &m2).unwrap();
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zd2c.png", temp_mat).unwrap();
    }
    if debug_info.edges_mat {
        //println!("PIXEL AT HSV (203, 1123): {:?}", hsv_mat.at_2d::<Vec3b>(1123, 203)?);
        //println!("PIXEL AT WRITE (203, 1123): {:?}", write_mat.at_2d::<Vec3b>(1123, 203)?);
        //println!("OLD PIXEL AT TEMP (203, 1123): {:?}", temp_mat.at_2d::<Vec3b>(1123, 203)?);
    }
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
    if debug_info.edges_mat {
        //println!("NEW PIXEL AT TEMP (203, 1123): {:?}", temp_mat.at_2d::<Vec3b>(1123, 203)?);
    }
    // Get the stuff that clearly is not goog
    // 30 * 255 / 100 = 76.5; HSV is (0, 0, 0) - (180, 100, 100),
    // but opencv represents the max as (180, 255, 255); OR
    // (255, 255, 255) if using HSV_FULL
    // I FOUND THE ISSUE
    in_range(
        temp_mat,
        &Vec3b::all(0),
        &Vec3b::from([5, 50, 80]),
        write_mat
    )?;
    // Make sure the pixels to dilate are colored
    bitwise_and(write_mat, write_mat, temp_mat, thresh_mat)?;
    if debug_info.edges_mat {
        //println!("VALUE PIXEL AT TEMP (203, 1123): {:?}", temp_mat.at_2d::<u8>(1123, 203)?);
    }
    if debug_info.debug_save {
        //imgproc::cvt_color(
        //    &temp_mat, 
        //    &mut m,
        //    imgproc::COLOR_HSV2BGR,
        //    0,
        //    AlgorithmHint::ALGO_HINT_DEFAULT
        //)?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zd3.png", temp_mat).unwrap();
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zd3b.png", write_mat).unwrap();
    }
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
    bitwise_and(write_mat, write_mat, hsv_mat, temp_mat)?;
    if debug_info.debug_save {
        //imgproc::cvt_color(
        //    &temp_mat, 
        //    &mut m,
        //    imgproc::COLOR_HSV2BGR,
        //    0,
        //    AlgorithmHint::ALGO_HINT_DEFAULT
        //)?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_mk.png", hsv_mat).unwrap();
    }
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
    // To get the most accurate picture, make sure every edge we detect is in
    // the original bit mat
    //extract_channel(
    //    hsvthresh_mat,
    //    hsv_mat,
    //    2
    //)?;
    //bitwise_and_def(hsv_mat, temp_mat, write_mat)?;
    if debug_info.debug_save {
        let mut m = Mat::default();
        bit_mat.copy_to_masked(&mut m, temp_mat)?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_pd.png", temp_mat).unwrap();
    }
    let t3 = std::time::Instant::now();
    if debug_info.print_phases {
        println!("Getting the grid Took: {} seconds", (t3 - t2).as_secs_f32());
    }
    let mut contours = Vector::<Vector::<Point>>::new();
    // get the contours of the processed image
    imgproc::find_contours(
        temp_mat,
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
    // We want to have the map contour index to search for warps and eliminate
    // points not in
    let mut biggest_contour_index = 0;
    let mut biggest_contour_area = 0.;
    let mut clusters = Vec::<Cluster>::with_capacity(contours.len());
    for (i, elem) in contours.iter().enumerate() {
        // (https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html#:~:text=From%20this%20moments,M%5B%27m00%27%5D.)
        let moments = imgproc::moments(&elem, true)?;
        let area = moments.m00;
        if area == 0. { continue }
        if area > biggest_contour_area {
            biggest_contour_area = area;
            biggest_contour_index = i;
        }
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
    let map = contours.get(biggest_contour_index)?;
    let t4 = std::time::Instant::now();
    if debug_info.print_phases {
        println!("Countour finding Took: {} seconds", (t4 - t3).as_secs_f32());
    }
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
        // make sure it's in
        let centr = Point2f::new(
            clusters[i].center.x as f32,
            clusters[i].center.y as f32
        );
        let testres = imgproc::point_polygon_test(
            &map,
            centr,
            false
        )?;
        // not inside
        if testres <= 0. { continue }
        let color = color_from_h(
            (i * 12) as f64 * 360. / clusters.len() as f64
        );
        let contour = contours.get(clusters[i].index)?;
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
            clusters[i].index,
            false
        ))));
    }
    let t5 = std::time::Instant::now();
    if debug_info.print_phases {
        println!("Cells Initial Took: {} seconds", (t5 - t4).as_secs_f32());
    }
    // Edge dilation adjacency check: (semilogs)
    if debug_info.dilate_contours && let Some(contour_mat) = &mut contour_mat {
        let mut a = Mat::default();
        imgproc::dilate_def(contour_mat, &mut a, closeby_kernel)?;
        *contour_mat = a;
    }
    for i in 0..cells.len() {
        for j in (i + 1)..cells.len() {
            // We know for sure they won't be the same cell as j != i
            if Cell::are_neighbors(&cells[i], &cells[j])
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
                write_mat,
                &mut contours,
                cells[i].borrow().index as i32,
                Scalar::all(255.),
                1,
                imgproc::LINE_8,
                &no_array(),
                i32::MAX,
                roi.tl() * -1
            )?;
            imgproc::dilate_def(write_mat, dilated_mat, neighbor_kernel)?;
            imgproc::draw_contours(
                temp_mat,
                &mut contours,
                cells[j].borrow().index as i32,
                Scalar::all(255.),
                1,
                imgproc::LINE_8,
                &no_array(),
                i32::MAX,
                roi.tl() * -1
            )?;
            bitwise_and_def(dilated_mat, temp_mat, and_mat)?;
            // Only count as neighbors if enough edge overlap exists:
            // count_non_zero(and_mat) >= 2*DILATION_PIXELS ensures real shared
            // edges pass while tiny corner/diagonal contacts with few
            // pixels are ignored
            if count_non_zero(and_mat)? >= 2 * DILATION_PIXELS {
                Cell::add_neighbor(
                    &cells[i],
                &cells[j]
                );
                continue
            }
            // Also want to see if it's closeby
            imgproc::dilate_def(write_mat, dilated_mat, closeby_kernel)?;
            bitwise_and_def(dilated_mat, temp_mat, and_mat)?;
            if count_non_zero(and_mat)? >= 2 * CLOSEBY_DILATION_PIXELS {
                // delete when debugging is done
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
    let t6 = std::time::Instant::now();
    if debug_info.print_phases {
        println!("Edge detection Took : {} seconds", (t6 - t5).as_secs_f32());
    }
    // Then add a step here that re-evaluates each cells neighbor
    let _ = &cells;
    // Then add a step here that turns the board into a petgraph
    let _ = &cells;
    // Also need to get rid of cells not part of the big cluster
    let mut dot_registry = Vec::<(Vec3b, [Location; 2])>::new();
    println!("C {}", cells.len());
    for cell in &cells {
        if cell.borrow().is_dot {
            let mut cell = cell.borrow_mut();
            if let Some((indx, (_, arr))) = dot_registry.iter_mut().enumerate()
            .find(|(_, x)| x.0 == cell.color) {
                let location = cell.center.into();
                // arr[0] is guaranteed to exist
                if arr[1].is_dne() {
                    arr[1] = location;
                    cell.affiliation = indx + 1;
                } else {
                    // We somehow have more than 2 cells with one color.
                    //return Err(opencv::Error::new(
                    //    0,
                    //    format!("{arr:?} and {location:?} all have the color {:?}", cell.color)
                    //))
                }
            } else {
                let aff = affiliation_displays.len() + 1;
                affiliation_displays.push(cell.color);
                cell.affiliation = aff;
                dot_registry.push((cell.color, [cell.center.into(), Location::dne()]));
            }
        }
        let cell = cell.borrow();
        // This should be enough to detect a windmill, but if not, consider looking at the vectors
        // or directions from the nodes to the center
        'windmill: {
            if (cell.neighbors.len() != 2 || cell.closeby.len() != 4) { break 'windmill }
            let mut points = [Point::default(); 4];
            let mut exit = false;
            // each closeby can only have exactly one neighbor in a windmill
            let rcs: [&Rc::<RefCell<Cell>>; 4] = std::array::from_fn(|i| {
                let rc = &cell.closeby[i];
                let closeby = rc.borrow();
                if closeby.neighbors.len() == 1 && closeby.closeby.len() == 1 {
                    exit = true;
                    points[i] = closeby.center;
                }
                drop(closeby);
                rc
            });
            if exit { break 'windmill }
            // Now we need to pair the slopes
            let mut zerospair = 255;
            let mut other1 = 255;
            let mut other2 = 255;
            for i in 1..4 {
                // 0 1 -> 2 3 -> (2 - (1 + 1) / 3) (3 - 1 / 3)
                // 0 2 -> 1 3 -> (2 - (1 + 2) / 3) (3 - 2 / 3)
                // 0 3 -> 1 2 -> (2 - (1 + 3) / 3) (3 - 3 / 3)
                other1 = 2 - (1 + i) / 3;
                other2 = 3 - i / 3;
                if approx_equal_slopes(points[0], points[i], cell.center)
                && approx_equal_slopes(points[other1], points[other2], cell.center)
                {
                    zerospair = i;
                    break;
                }
            }
            if zerospair == 255 { break 'windmill }
            Cell::add_neighbor(rcs[0], rcs[zerospair]);
            Cell::add_neighbor(rcs[other1], rcs[other2]);
        }
        'bridge: {
 
        }
        'alley: {

        }
        'overpass: {

        }
        'warp: {

        }
        // I just realized we don't need special processing for chains because
        // all but 2 termini will be right by each other, so it doesn't matter
        // for **solving**, but when **drawing**, we have to make sure we go to
        // the neighboring dot without releasing
        // Just checked, it doesn't even matter for drawing! The "chains" are
        // purely visual
        // 'chain: { }
    }
    let t7 = std::time::Instant::now();
    if debug_info.print_phases {
        println!("Graph Creation Took: {} seconds", (t7 - t6).as_secs_f32());
    }
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
    // 4 times is probably a good estimate
    let mut graph = UnGraphMap::<GraphCell, GraphEdge>::with_capacity(
        cells.len(),
        4 * cells.len()
    );
    for cell in &cells {
        let cell = cell.borrow();
        let gcell = GraphCell::new_def(
            cell.affiliation, cell.center.into(), cell.is_dot, cell.is_phantom
        );
        graph.add_node(gcell);
        for neighbor in &cell.neighbors {
            let neighbor = neighbor.borrow();
            let gneighbor = GraphCell::new_def(
                neighbor.affiliation, neighbor.center.into(), neighbor.is_dot,
                neighbor.is_phantom
            );
            graph.add_node(gneighbor);
            let res = graph.add_edge(gcell, gneighbor, GraphEdge { affiliation: 0 });
            // This is a new edge
            if res.is_none() {
                if let Some(final_mat) = &mut arrow_mat {
                    imgproc::arrowed_line(
                        final_mat,
                        cell.center,
                        neighbor.center,
                        Scalar::all(255.),
                        3,
                        imgproc::LINE_8,
                        0,
                        0.1
                    )?;
                }
            }
        }
    }
    println!("NUM IN GRID: {}", graph.node_count());
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
    return Ok((graph, dmats));
}