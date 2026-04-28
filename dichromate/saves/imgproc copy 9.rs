use std::{cell::RefCell, f64::consts::PI, rc::Rc};
use itertools::Itertools;
use opencv::{
    core::*, imgproc::*,
};
use permanganate::arbitrary::{Coordinates, GraphCell, GraphCellHint, GraphEdge};
use petgraph::prelude::UnGraphMap;
use rayon::prelude::*;

const DILATION_PIXELS: i32 = 4;

pub fn stats_with_mask(
    channel: &Mat,
    mask: &Mat,
) -> Result<(f64, u8, u8, f64), opencv::Error> {
    let data = channel.data_bytes()?; // assumes continuous
    let msk = mask.data_bytes()?;

    let mut hist = [0u32; 256];
    let mut sum: u64 = 0;
    let mut sum_sq: u64 = 0;
    let mut count: u32 = 0;

    // Single pass: histogram + sum + sum of squares
    for i in 0..data.len() {
        if msk[i] != 0 {
            let v_u64 = data[i] as u64;
            hist[v_u64 as usize] += 1;
            sum += v_u64;
            sum_sq += v_u64 * v_u64;
            count += 1;
        }
    }

    if count == 0 {
        return Ok((0., 0, 0, 0.)); // mean, median, mode, stddev
    }

    // ---- Mean ----
    let mean = sum as f64 / count as f64;

    // ---- Median ----
    let mid = count / 2;
    let mut acc = 0u32;
    let mut median = 0;

    for i in 0..256 {
        acc += hist[i];
        if acc > mid {
            median = i as u8;
            break;
        }
    }

    // ---- Mode ----
    let mut mode = 0;
    let mut max_count = 0;

    for i in 0..256 {
        if hist[i] > max_count {
            max_count = hist[i];
            mode = i as u8;
        }
    }

    // ---- Standard Deviation ----
    let mean_sq = sum_sq as f64 / count as f64;
    let variance = mean_sq - (mean * mean);
    let stddev = variance.sqrt();
    Ok((mean, median, mode, stddev))
}

#[derive(Debug, Clone)]
pub struct Cluster {
    area: f64,
    index: usize,
    center: Point,
    is_bad: bool,
}

pub struct AffiliationManager {
    pub displays: Vec<Vec3b>,
    pub affiliations: Vec<usize>,
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
    cross.abs() <= 100. //1e-12f64.max(1e-9f64 * max_mag)
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

struct BridgeCmd {
    px: i32,
    py: i32,
    dx: f32,
    dy: f32,
    value: u8,
}

// DDA-based ray walker
fn walk_ray(
    data: &[u8],
    cols: i32,
    rows: i32,
    start_x: i32,
    start_y: i32,
    dx: f32,
    dy: f32,
    max_dist: i32,   // Chebyshev distance
    stop_on_zero: bool,
) -> i32 {
    let mut fx = start_x as f32;
    let mut fy = start_y as f32;

    let mut last_x = start_x;
    let mut last_y = start_y;

    let mut count = 0;

    // large upper bound just to prevent infinite loops
    for _ in 0..(2 * max_dist + 10) {
        fx += dx;
        fy += dy;

        let x = fx.round() as i32;
        let y = fy.round() as i32;

        if x == last_x && y == last_y { continue; }
        last_x = x;
        last_y = y;

        if x < 0 || x >= cols || y < 0 || y >= rows {
            break;
        }

        // Stop when the ray leaves the square
        if (x - start_x).abs() > max_dist || (y - start_y).abs() > max_dist {
            break;
        }

        let idx = (y * cols + x) as usize;

        if data[idx] != 0 {
            count += 1;
        } else if stop_on_zero {
            break;
        }
    }

    count
}
// Main function with θ-based rays
fn bridge_gaps_single_pass(
    bw: &mut Mat,
    max_back: i32,
    min_run: i32,
    max_forward: i32,
    forward_presence_radius: i32,
    directions: &[(f32, f32)], // theta directions as (cos, sin)
) -> opencv::Result<()> {
    let rows = bw.rows();
    let cols = bw.cols();
    let data = bw.data_bytes_mut()?;

    let start = std::time::Instant::now();
    let bridge_commands: Vec<BridgeCmd> = (0..rows)
        .into_par_iter()
        .map(|y| {
            let mut local_cmds = Vec::with_capacity(32);

            for x in 0..cols {
                let idx = (y * cols + x) as usize;
                if data[idx] != 0 {
                    continue;
                }

                let mut best_run = -1;
                let mut best_dir = (1., 1.);

                for &(dx, dy) in directions {
                    let run = walk_ray(
                        data, cols, rows, x, y, dx, dy,
                        max_back, true
                    );
                    if run > best_run {
                        best_run = run;
                        best_dir = (dx, dy);
                    }
                }
                if best_run < min_run {
                    continue;
                }
                let (dx, dy) = best_dir;
                let found = walk_ray(
                    data, cols, rows, x, y, -dx, -dy,
                    forward_presence_radius, false,
                ) > 0;
                
                if !found {
                    continue;
                }

                // 3. Valid direction
                local_cmds.push(BridgeCmd {
                    px: x,
                    py: y,
                    dx,
                    dy,
                    value: 255,
                });
            }

            local_cmds
        })
        .reduce(Vec::new, |mut a, b| {
            a.extend(b);
            a
        });
    println!("PaR TOOL : {}", (std::time::Instant::now() - start).as_secs_f32());
    // APPLY PHASE: draw inverse ray along −θ
    for cmd in bridge_commands {
        let mut fx = cmd.px as f32;
        let mut fy = cmd.py as f32;

        let mut last_x = cmd.px;
        let mut last_y = cmd.py;

        for step in 0..max_forward {
            fx -= cmd.dx;
            fy -= cmd.dy;

            let nx = fx.round() as i32;
            let ny = fy.round() as i32;

            if nx == last_x && ny == last_y {
                continue;
            }
            last_x = nx;
            last_y = ny;

            if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
                break;
            }

            let idx = (ny * cols + nx) as usize;

            if data[idx] != 0 && step != 0 {
                break;
            }

            data[idx] = cmd.value;
        }
    }

    Ok(())
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

pub struct CIELCH {
    gamma_lut: [f32; 256],
    lstar_lut: [u8; 4096],
}

impl CIELCH {
    pub fn new() -> Self {
        Self {
            gamma_lut: Self::build_srgb_to_linear_lut(),
            lstar_lut: Self::build_lstar_lut(),
        }
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

    fn build_lstar_lut() -> [u8; 4096] {
        let size = 4096;
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
        &self,
        bgr: &[u8],
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

                    let r_lin = self.gamma_lut[bgr[idx + 2] as usize];
                    let g_lin = self.gamma_lut[bgr[idx + 1] as usize];
                    let b_lin = self.gamma_lut[bgr[idx] as usize];

                    let y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin;
                    let lut_idx = (y * 4095.0) as usize;

                    *out_px = self.lstar_lut[lut_idx];
                }
            });
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
    slight_kernel: Mat,
    border_value: Scalar,
    affiliation_displays: Vec<Vec3b>,
    lightness: Vec<u8>,
    gamma_lut: [f32; 256],
    lstar_lut: [u8; 4096],
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
            circle_kernel: get_structuring_element(
                MORPH_ELLIPSE,
                Size::new(5, 5),
                Point::new(-1, -1)
            )?,
            // kernel to dilate and erode the grid by to connect broken lines
            extend_kernel: get_structuring_element(
                MORPH_RECT,
                Size::new(5, 5),
                Point::new(-1, -1)
            )?,
            neighbor_kernel: get_structuring_element(
                MORPH_ELLIPSE,
                Size::new(2 * DILATION_PIXELS + 1, 2 * DILATION_PIXELS + 1),
                Point::new(-1, -1)
            )?,
            // For very small pixel operations
            slight_kernel: get_structuring_element(
                MORPH_ELLIPSE, Size::new(3, 3),
                Point::new(-1, -1)
            )?,
            // Default border value
            border_value: morphology_default_border_value()?,
            // Our internal reference for terminal nodes
            affiliation_displays: Vec::new(),
            // Array to store CIE LAB Lightness of pixels
            lightness: Vec::new(),
            // Look up tables for conversion from bgr to L (AB are not found)
            gamma_lut: build_srgb_to_linear_lut(),
            lstar_lut: build_lstar_lut(),
        })
    }

    pub fn get_affiliations(&self) -> &Vec<Vec3b> {
        return &self.affiliation_displays;
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

    // Code must convert the original mat to hsv_full
    pub fn detect_cells(
        &mut self,
        bit_mat: &Mat,
        debug_info: &DebugInfo,
        code: i32
    ) -> opencv::error::Result<(UnGraphMap<GraphCell, GraphEdge>, DetectedMats)> {
        self.clean()?;
        detect_cells(
            &bit_mat,
            &debug_info,
            &self.circle_kernel,
            &self.extend_kernel,
            &self.neighbor_kernel,
            self.border_value,
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
            &self.lstar_lut,
            code
        )
    }

    // Plan: for chain maps, let the 2 cells become one, but try to undo them
    // as in, if it's big and we can split into 2 and we'd have 2 dots if we did
    // indeed go forth with this action, then proceed gng
    pub fn detect_cells2(
        &mut self,
        bit_mat: &Mat,
        debug_info: &DebugInfo,
        bgr: bool
    ) -> opencv::Result<Mat> {
        let start = std::time::Instant::now();
        // We only have these 2 codes for converting to opencv
        let code = if bgr { COLOR_BGR2HSV_FULL } else { COLOR_RGB2HSV_FULL };
        let hsv_mat = convert_to_hsv(&bit_mat, code)?;
        let (lowerthresh, middlethresh, upperthresh) = create_thresholds(&hsv_mat)?;
        let (hsv_lower, hsv_mid, hsv_upper) = apply_thresholds(
            &hsv_mat, &lowerthresh, &middlethresh, &upperthresh
        )?;
        // We get the circles from the mat and we dilate them to cover a larger area
        // The reason for this is because of the rings that appear around endpoint dots
        // in chain levels.
        let dilated_circles = open_and_dilate(&hsv_upper, &self.circle_kernel, self.border_value)?;
        let [nhsv, hue_alt, inverted_features, non_zero] = remove_and_isolate_hue(&hsv_upper, &upperthresh, &dilated_circles, &self.circle_kernel, self.border_value)?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/U11.png", &nhsv).unwrap();
        // hsv_alt is guaranteed to have the circles and rings removed. BUT it might get rid of
        // some pixels of the grid, so we need to recover lost pixels
        let stats = stats_with_mask(&hue_alt, &non_zero)?;
        //opencv::imgcodecs::imwrite_def("dichromate/pics2/U5.png", &upperthresh).unwrap();
        opencv::imgcodecs::imwrite_def("dichromate/pics2/U6.png", &non_zero).unwrap();
        println!("STATS: {stats:?}");
        // The board without dots or bg (some dots might have the same hue so remove them)
        let bare_board = apply_stats(stats, &hsv_mid, &non_zero, &inverted_features)?;
        let new_features = get_newly_added_features(&non_zero, &bare_board)?;
        let full_board = delete_select_features(&new_features, &bare_board, &self.slight_kernel, self.border_value)?;
        let final_prod = connect_detached(&full_board, &self.extend_kernel, self.border_value)?;
        // What wasn't there before
        opencv::imgcodecs::imwrite_def("dichromate/pics2/U2.png", &bare_board).unwrap();
        opencv::imgcodecs::imwrite_def("dichromate/pics2/U7.png", &new_features).unwrap();
        opencv::imgcodecs::imwrite_def("dichromate/pics2/U17.png", &final_prod).unwrap();
        println!("{:?}", stats);
        println!("{}", (std::time::Instant::now() - start).as_secs_f32());

        Ok(final_prod)
    }
}

fn convert_to_hsv(bit_mat: &Mat, code: i32) -> opencv::Result<Mat> {
    let mut hsv_mat = Mat::default();
    cvt_color(
        bit_mat, 
        &mut hsv_mat,
        code,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT
    )?;
    return Ok(hsv_mat);
}

fn create_thresholds(hsv_mat: &Mat) -> opencv::Result<(Mat, Mat, Mat)> {
    let mut lowerthresh = Mat::default();
    let mut middlethresh = Mat::default();
    let mut upperthresh = Mat::default();

    // This is the "base" threshold, everything removed from here is guaranteed
    // to be something that we don't want
    let mut remove = Mat::default();
    in_range(
        hsv_mat,
        &Vec3b::from_array([0, 0, 0]),
        &Vec3b::from_array([255, 255, 4]), // 17
        &mut remove
    )?;
    bitwise_not_def(&remove, &mut lowerthresh)?;

    /* This is the middle threshold. This might not have everything we want,
       and it might not get rid of everything we don't want, but it does
       get rid of enough for us to move on  */
    let mut keep = Mat::default();
    let mut keep2 = Mat::default();
    let mut keep3 = Mat::default();
    let mut combined1 = Mat::default();
    // 1. Find pixels with lower L values (CIE LAB)
    //threshold(
    //    &lmat,
    //    &mut keep,
    //    15.,
    //    255.,
    //    THRESH_BINARY_INV
    //)?;
    // 2. find pixels with low S values
    in_range(
        hsv_mat,
        &Vec3b::from_array([0, 0, 0]),
        &Vec3b::from_array([255, 255, 15]),
        &mut remove
    )?;
    // 3. Invert it (we don't want pixel with low V values)
    bitwise_not_def(&remove, &mut keep2)?;
    // 4. Get blobs (most are characterized by high saturation and low value)
    in_range(
        &hsv_mat,
        &Vec3b::from([0, 225, 0]),
        &Vec3b::from([255, 255, 90]),
        &mut remove
    )?;
    // 5. Invert it (we don't want blob pixels)
    bitwise_not_def(&remove, &mut keep3)?;
    // 6. We want pixels that pass all 3 tests
    combined1 = keep2;
    //bitwise_and_def(&keep, &keep2, &mut combined1)?;
    bitwise_and_def(&combined1, &keep3, &mut keep)?;
    in_range(
        &hsv_mat,
        &Vec3b::from([0, 0, 45]),
        &Vec3b::from([255, 255, 255]),
        &mut middlethresh
    )?;    
    // This is the upperthresh. A harsher lowerthresh.
    in_range(
        &hsv_mat,
        &Vec3b::from([0, 0, 0]),
        &Vec3b::from([255, 255, 84]), // floor (33 * 2.55)
        &mut remove
    )?;
    bitwise_not_def(&remove, &mut upperthresh)?;
    Ok((lowerthresh, middlethresh, upperthresh))
}

fn apply_thresholds(hsv_mat: &Mat, lowerthresh: &Mat, middlethresh: &Mat, upperthresh: &Mat) -> opencv::Result<(Mat, Mat, Mat)> {
    let mut hsv_lower = Mat::default();
    let mut hsv_middle = Mat::default();
    let mut hsv_upper = Mat::default();

    hsv_mat.copy_to_masked(&mut hsv_lower, &lowerthresh)?;
    hsv_mat.copy_to_masked(&mut hsv_middle, &middlethresh)?;
    hsv_mat.copy_to_masked(&mut hsv_upper, &upperthresh)?;
    Ok((hsv_lower, hsv_middle, hsv_upper))
}

fn open_and_dilate(hsv_upper: &Mat, circle_kernel: &Mat, border_value: Scalar) -> opencv::Result<Mat> {
    let mut circles = Mat::default();
    let mut dilated_circles = Mat::default();
    morphology_ex(
        hsv_upper,
        &mut circles,
        MORPH_OPEN,
        circle_kernel,
        Point::new(-1, -1),
        4,
        BORDER_CONSTANT,
        border_value,
    )?;
    morphology_ex(
        &circles,
        &mut dilated_circles,
        MORPH_DILATE,
        circle_kernel,
        Point::new(-1, -1),
        7,
        BORDER_CONSTANT,
        border_value,
    )?;
    Ok(dilated_circles)
}

fn remove_and_isolate_hue(hsv_upper: &Mat, upperthresh: &Mat, dilated_circles: &Mat, circle_kernel: &Mat, border_value: Scalar) -> opencv::Result<[Mat; 4]> {
    let mut diff = Mat::default();
    let mut hdiff = Mat::default();
    let mut hwrap = Mat::default();
    let mut hmin = Mat::default();
    let mut thresh1 = Mat::default();
    let mut thresh2 = Mat::default();
    let mut dilated = Mat::default();
    let mut inverted_features = Mat::default();
    let mut hue_component = Mat::default();
    let mut hsv_reformed = Mat::default();
    let mut dst = Mat::default();
    // We want to remove the circle (+ rings if they exist)
    // Absdiff so where the circles and rings are will be black
    absdiff(hsv_upper, dilated_circles, &mut diff)?;
    // We need to sort of encode the cyclic nature of hsv
    // absdiff(255, 1) = 254, but it should be 1 in hsv
    // min(|a - b|, 256 - |a - b|)
    extract_channel(&diff, &mut hdiff, 0)?;
    subtract_def(
        &Scalar::all(256.),
        &hdiff,
        &mut hwrap,
    )?;
    min(&hdiff, &hwrap, &mut hmin)?;
    insert_channel(&hmin, &mut diff, 0)?;
    // Get pixels with about 0 hue (note this also includes black pixels)
    in_range(
        &diff,
        &Vec3b::all(0),
        &Vec3b::from([3, 50, 80]),
        &mut thresh1
    )?;
    // Make sure the pixels to remove / dilate are colored
    bitwise_or(&thresh1, &thresh1, &mut thresh2, upperthresh)?;
    // Usually, the very edges of the dot / ring aren't in the range, so we
    // dilate the area to remove just a tiny bit extra to make sure we get everything
    dilate(
        &thresh2,
        &mut dilated,
        circle_kernel,
        Point::new(-1, -1),
        2,
        BORDER_CONSTANT,
        border_value
    )?;
    // Invert it so that we know where to not copy
    bitwise_not_def(&dilated, &mut inverted_features)?;
    opencv::imgcodecs::imwrite_def("dichromate/pics2/U3.png", &inverted_features).unwrap();
    //extract_channel(hsv_upper, &mut hue_component, 0)?;
    {
        let mut k = Mat::default();
        cvt_color(&hsv_upper, &mut k, COLOR_HSV2BGR_FULL, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/U5.png", &k).unwrap();
    };
    //opencv::imgcodecs::imwrite_def("dichromate/pics2/U4.png", &hue_component).unwrap();
    bitwise_or(&hsv_upper, &hsv_upper, &mut hsv_reformed, &inverted_features)?;
    let mut a = Mat::default();
    hsv_upper.copy_to_masked(&mut a, &inverted_features)?;
    hsv_upper.copy_to_masked(&mut hsv_reformed, &inverted_features)?;
    opencv::imgcodecs::imwrite_def("dichromate/pics2/U1.png", &hsv_reformed).unwrap();
    extract_channel(&hsv_reformed, &mut hue_component, 0)?;
    opencv::imgcodecs::imwrite_def("dichromate/pics2/U16.png", &hue_component).unwrap();
    in_range(&hsv_reformed, &Vec3b::from_array([0, 0, 1]), &Vec3b::from_array([255, 255, 255]),  &mut dst)?;
    Ok([hsv_reformed, hue_component, inverted_features, dst])
}

fn apply_stats(
    (mean, median, mode, stddev): (f64, u8, u8, f64),
    hsv_lower: &Mat,
    non_zero: &Mat,
    inverted_features: &Mat,
) -> opencv::Result<Mat> {
    let mut keep = Mat::default();
    let mut lvl = Mat::default();
    // Because we could be looking at the entire flow game screen, there is a
    // non zero chance that the white in the picture slightly exceeds the actual
    // hue of the board (because technically the board uses a hue range). Mean
    // Do not immediately clamp. Red has a hue of ~0.
    let (lower_hue, lof) = median.overflowing_sub(7);
    let (upper_hue, uof) = median.overflowing_add(7);
    if !lof && !uof  {
        println!("NO OVERFLOW");
        in_range(
            hsv_lower,
            &Vec3b::from_array([lower_hue, 20, 20]),
            &Vec3b::from_array([upper_hue, 255, 255]),
            &mut keep
        )?;
    } else if lof && uof {
        return Err(opencv::Error::new(1, "Yo ts ain't posssible gng. How"));
    } else {
        println!("OVERFLOW");
        let mut keep2 = Mat::default();
        in_range(
            hsv_lower,
            &Vec3b::from_array([0, 20, 20]),
            &Vec3b::from_array([upper_hue, 255, 255]),
            &mut keep2
        )?;
        in_range(
            hsv_lower,
            &Vec3b::from_array([lower_hue, 20, 20]),
            &Vec3b::from_array([255, 255, 255]),
            &mut lvl
        )?;
        bitwise_or(&keep2, &lvl, &mut keep, &inverted_features)?;
    }
    bitwise_or(&keep, &non_zero, &mut lvl, &inverted_features)?;
    Ok(lvl)
}

fn get_newly_added_features(nhsv: &Mat, bare_board: &Mat) -> opencv::Result<Mat> {
    // fiind what is in bare, but not in nshv
    let mut a = Mat::default();
    let mut b = Mat::default();
    bitwise_not_def(&nhsv, &mut a)?;
    bitwise_and_def(&bare_board, &a, &mut b)?;
    Ok(b)
}

// This function was originally fairly simple, but it had one issue. Extreme
// alleys. In extreme alleys, there can be dim thick line behind a bright thin
// line, so when we do the hue recovery, it'll see the dim thick line as 2 thinner
// dim lines. Now this was a problem because in extreme alleys, only God knows
// why, whenever a grid line of thin bright pixels connect to a thick bright pixels,
// they turn dim and borderline invisible. So when this function would not delete
// those dim thin lines connected to a thick, they would also not delete the now
// dim thin likes that were behind a bright thin lines because they basically
// have the same level of thickness. So instead of just a simple open + dilate,
// I have to it again. The first open get the "significant" new features. the
// dilate enhances them, with the goal of connecting the 2 parallel thin lines
// into one thick line if they exist. Now we open again to get rid of any
// single thin lines, then we diilate because open might've removed some pixels
// that we want to capture. then we bitwise and with the original new_features
// so we only delete new features and nothing extra. It doesn't perfectly work,
// but it's good enough for now.
fn delete_select_features(new_features: &Mat, bare_board: &Mat, slight_kernel: &Mat, border_value: Scalar) -> opencv::Result<Mat> {
    let mut dst = Mat::default();
    let mut inv = Mat::default();
    let mut dst2 = Mat::default();
    let mut fin = Mat::default();

    let mut a = Mat::default();
    let mut b = Mat::default();
    let mut c = Mat::default();
    let mut d = Mat::default();

    morphology_ex(
        new_features,
        &mut dst,
        MORPH_OPEN,
        slight_kernel,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        border_value,
    )?;
    morphology_ex(
        &dst,
        &mut a,
        MORPH_DILATE,
        slight_kernel,
        Point::new(-1, -1),
        3,
        BORDER_CONSTANT,
        border_value,
    )?;
    morphology_ex(
        &a,
        &mut b,
        MORPH_OPEN,
        slight_kernel,
        Point::new(-1, -1),
        6,
        BORDER_CONSTANT,
        border_value,
    )?;
    morphology_ex(
        &b,
        &mut c,
        MORPH_DILATE,
        slight_kernel,
        Point::new(-1, -1),
        3,
        BORDER_CONSTANT,
        border_value,
    )?;
    bitwise_and_def(&dst, &c, &mut d)?;
    opencv::imgcodecs::imwrite_def("dichromate/pics2/U12.png", &a).unwrap();
    opencv::imgcodecs::imwrite_def("dichromate/pics2/U13.png", &b).unwrap();
    opencv::imgcodecs::imwrite_def("dichromate/pics2/U14.png", &c).unwrap();
    opencv::imgcodecs::imwrite_def("dichromate/pics2/U15.png", &d).unwrap();
    // delete from of
    bitwise_not_def(&d, &mut inv)?;
    bare_board.copy_to_masked(&mut dst2, &inv)?;
    // the slightest erosion
    let slight_kernel = get_structuring_element(
        MORPH_ELLIPSE, Size::new(2, 2), Point::new(-1, -1)
    )?;
    morphology_ex(
        &dst2,
        &mut fin,
        MORPH_OPEN,
        &slight_kernel,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        border_value,
    )?;
    opencv::imgcodecs::imwrite_def("dichromate/pics2/U8.png", &dst).unwrap();
    opencv::imgcodecs::imwrite_def("dichromate/pics2/U9.png", &dst2).unwrap();
    opencv::imgcodecs::imwrite_def("dichromate/pics2/U10.png", &fin).unwrap();
    Ok(fin)
}

fn connect_detached(full_board: &Mat, extend_kernel: &Mat, border_value: Scalar) -> opencv::Result<Mat> {
    let mut res = Mat::default();
    morphology_ex(
        &full_board,
        &mut res,
        MORPH_CLOSE,
        &extend_kernel,
        Point::new(-1, -1),
        3,
        BORDER_CONSTANT,
        border_value,
    )?;
    Ok(res)
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
    border_value: Scalar,
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
    lstar_lut: &[u8; 4096],
    code: i32
) -> opencv::error::Result<(UnGraphMap<GraphCell, GraphEdge>, DetectedMats)> {
    let mut dmats = DetectedMats::default();
    // Convert to hsv (store in a temporary mat, we need to do some processing)
    cvt_color(
        bit_mat, 
        temp_mat,
        code,
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
    threshold(
        &lmat,
        blob_mat,
        15.,
        255.,
        THRESH_BINARY_INV
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
    threshold(
        thresh_mat,
        write_mat,
        15.,
        255.,
        THRESH_BINARY_INV
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
    // hsv mat with all the noise gone // (hsv mat is zeroed)
    temp_mat.copy_to_masked(hsv_mat, thresh_mat)?;
    let keep = hsv_mat.clone();
    // color mat with all the noise gone // (colorthresh mat is zeroed)
    bit_mat.copy_to_masked(colorthresh_mat, thresh_mat)?;
    if debug_info.debug_save {
        let mut m = Mat::default();
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zct.png", colorthresh_mat).unwrap();
    }
    let mut arrow_mat = debug_info.arrow_mat.then(|| bit_mat.clone());
    if debug_info.edges_mat {
        dmats.edges_mat = Some(thresh_mat.clone());
    }
    // Isolate the circles
    morphology_ex(
        hsv_mat,
        temp_mat,
        MORPH_OPEN,
        circle_kernel,
        Point::new(-1, -1),
        4,
        BORDER_CONSTANT,
        border_value,
    )?;
    //opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_hsv.png", hsv_mat).unwrap();
    if debug_info.debug_save {
        let mut m = Mat::default();
        //opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zctp.png", &x).unwrap();
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zct2.png", temp_mat).unwrap();
    }
    if debug_info.debug_save {
        let mut m = Mat::default();
        cvt_color(
            temp_mat, 
            &mut m,
            COLOR_HSV2BGR,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT
        )?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zc.png", &m).unwrap();
    }
    // dilate the circles (3 first, remove blobs, then 4)
    morphology_ex(
        temp_mat,
        write_mat,
        MORPH_DILATE,
        circle_kernel,
        Point::new(-1, -1),
        3,
        BORDER_CONSTANT,
        border_value,
    )?;
    // remove not (high saturation and high volume)
    in_range(
        write_mat,
        &Vec3b::from_array([0, 0, 100]),
        &Vec3b::from_array([255, 255, 255]),
        temp_mat
    )?;
    // stuff that is in temp, but not in write, must be removed from the original
    bitwise_not_def(temp_mat, blob_mat)?;
    // (temp mat is NOT zeroed)
    temp_mat.set_scalar(0.into())?;
    write_mat.copy_to_masked(temp_mat, blob_mat)?;
    threshold(
        temp_mat,
        blob_mat,
        1.,
        255.,
        THRESH_BINARY_INV
    )?;
    // (other mat is zeroed)
    hsv_mat.copy_to_masked(other_mat, blob_mat)?;
    std::mem::swap(hsv_mat, other_mat);
    // (temp mat NOT is zeroed)
    temp_mat.set_scalar(0.into())?;
    write_mat.copy_to_masked(temp_mat, blob_mat)?;
    if debug_info.debug_save {
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zcp0.png", temp_mat).unwrap();
    }
    // The last 4 dilations
    morphology_ex(
        temp_mat,
        write_mat,
        MORPH_DILATE,
        circle_kernel,
        Point::new(-1, -1),
        4,
        BORDER_CONSTANT,
        border_value,
    )?;
    if debug_info.debug_save {
        let mut m = Mat::default();
        cvt_color(
            write_mat, 
            &mut m,
            COLOR_HSV2BGR,
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
        cvt_color(
            temp_mat, 
            &mut m,
            COLOR_HSV2BGR_FULL,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT
        )?;
        cvt_color(
            hsv_mat, 
            &mut m2,
            COLOR_HSV2BGR_FULL,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT
        )?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zd2.png", &m).unwrap();
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zd2b.png", &m2).unwrap();
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zd2c.png", temp_mat).unwrap();
    }
    // Alright so in the hsv space, hue is circular, i.e. 255 and 0 are almost
    // the same hue. So imagine if the hue in the hsv_mat is 255, but it's 0
    // in the write_mat. Very bad. So we need to sort of encode the cyclic
    // nature of hsv
    extract_channel(temp_mat, blob_mat, 0)?;
    subtract_def(
        &Scalar::all(128.),
        blob_mat,
        write_mat,
    )?;
    min(blob_mat, write_mat, and_mat)?;
    insert_channel(and_mat, temp_mat, 0)?;
    // Get the stuff that clearly is not goog
    // 30 * 255 / 100 = 76.5; HSV is (0, 0, 0) - (180, 100, 100),
    // but opencv represents the max as (180, 255, 255); OR
    // (255, 255, 255) if using HSV_FULL
    // I FOUND THE ISSUE
    in_range(
        temp_mat,
        &Vec3b::all(0),
        &Vec3b::from([3, 50, 80]),
        write_mat
    )?;
    // Make sure the pixels to dilate are colored
    // (temp mat is NOT zeroed)
    temp_mat.set_scalar(0.into())?;
    write_mat.copy_to_masked(temp_mat, thresh_mat)?;
    if debug_info.debug_save {
        //cvt_color(
        //    &temp_mat, 
        //    &mut m,
        //    COLOR_HSV2BGR,
        //    0,
        //    AlgorithmHint::ALGO_HINT_DEFAULT
        //)?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zd3.png", temp_mat).unwrap();
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_zd3b.png", write_mat).unwrap();
    }
    // Usually, the very edges of the dot / ring aren't in the range, so we
    // dilate the area to remove just a tiny bit to make sure we get everything
    dilate(
        temp_mat,
        write_mat,
        circle_kernel,
        Point::new(-1, -1),
        2,
        BORDER_CONSTANT,
        border_value
    )?; 
    // Invert it so that we know where to not copy
    bitwise_not_def(write_mat, temp_mat)?;
    // We no longer need the full hsv mat. We only need the values now.
    extract_channel(
        hsv_mat,
        write_mat,
        2
    )?;
    {
        // kisnisd
        let mut a1 = Mat::default();
        let mut a2 = Mat::default();
        let mut a3 = Mat::default();
        let mut a4 = Mat::default();
        let mut a5 = Mat::default();
        let mut hue_mean = Scalar::all(0.);
        let mut hue_std = Scalar::all(0.);
        extract_channel(
            hsv_mat,
            &mut a1,
            0
        )?;
        a1.copy_to_masked(&mut a2, temp_mat)?;
        in_range(
            &a2,
            &Scalar::new(1., 0., 0., 0.),
            &Scalar::new(255., 255., 255., 255.),
            &mut a3,
        )?;
        if debug_info.debug_save {
            //cvt_color(
            //    &temp_mat, 
            //    &mut m,
            //    COLOR_HSV2BGR,
            //    0,
            //    AlgorithmHint::ALGO_HINT_DEFAULT
            //)?;
            opencv::imgcodecs::imwrite_def("dichromate/pics2/T_a1.png", &a3).unwrap();
        }
        mean_std_dev(&a2, &mut hue_mean, &mut hue_std, &a3)?;
        let med = stats_with_mask(&a2, &thresh_mat).unwrap();
        println!("MEAN: {hue_mean:?}, median: {med:?}, std dev: {hue_std:?}");
        let lower = hue_mean + Scalar::new(-10., 0., 0., 0.);
        let upper = hue_mean + Scalar::new(10., 215., 255., 255.);
        in_range(
            &keep,
            &lower,
            &upper,
            &mut a4,
        )?;
        if debug_info.debug_save {
            //cvt_color(
            //    &temp_mat, 
            //    &mut m,
            //    COLOR_HSV2BGR,
            //    0,
            //    AlgorithmHint::ALGO_HINT_DEFAULT
            //)?;
            opencv::imgcodecs::imwrite_def("dichromate/pics2/T_a2.png", &a4).unwrap();
        }
    }
    // (hsv mat NOT is zeroed)
    // Only copy pixels that aren't part of dots or rings
    hsv_mat.set_scalar(0.into())?;
    write_mat.copy_to_masked(hsv_mat, temp_mat)?;
    if debug_info.debug_save {
        //cvt_color(
        //    &temp_mat, 
        //    &mut m,
        //    COLOR_HSV2BGR,
        //    0,
        //    AlgorithmHint::ALGO_HINT_DEFAULT
        //)?;
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_mk.png", hsv_mat).unwrap();
    }
    // Some small segments of lines were erased during either the initial
    // threshold, or from the removal of dots and rings. Of the cases where
    // the lines are erased, the former makes up the majority. Dilate and erode
    // the image to bridge those small gaps
    // close(x, iter) = erode(dilate(x, iter), iter)
    // Unfortulately, morph close tends to make corners bigger than they should
    // be, but there's nothing we can do but hope it doesnt shoot us in the
    // foot
    morphology_ex(
        hsv_mat,
        temp_mat,
        MORPH_CLOSE,
        extend_kernel,
        Point::new(-1, -1),
        3,
        BORDER_CONSTANT,
        border_value,
    )?;
    // let directions: Vec<(f32, f32)> = (0..64)
    //        .map(|i| {
    //            let theta = (i as f32) * std::f32::consts::PI / 32.;
    //            (theta.cos(), theta.sin())
    //        })
    //        .collect();
    //bridge_gaps_single_pass(
    //    hsv_mat,
    //    40,
    //    35,
    //    50,
    //    20,
    //    &directions
    //)?;
    let mut hough: Vector<Vec4i> = Vector::new();
    hough_lines_p(
        hsv_mat,
        &mut hough,
        1.,
        PI / 180.,
        5,
        3.,
        25.,
    )?;
    let mut a = colorthresh_mat.clone();
    for l in hough {
        line(
            &mut a,
            Point::new(l[0], l[1]),
            Point::new(l[2], l[3]),
            Scalar::new(0.,0.,255., 255.),
            3,
            LINE_AA,
            0
        )?;
    }
    if debug_info.debug_save {
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_px.png", &a).unwrap();
    }
    //std::mem::swap(hsv_mat, temp_mat);
    if debug_info.debug_save {
        opencv::imgcodecs::imwrite_def("dichromate/pics2/T_gray_pd.png", temp_mat).unwrap();
    }
    let t3 = std::time::Instant::now();
    if debug_info.print_phases {
        println!("Getting the grid Took: {} seconds", (t3 - t2).as_secs_f32());
    }
    let mut contours = Vector::<Vector::<Point>>::new();
    // get the contours of the processed image
    find_contours(
        temp_mat,
        &mut contours,
        RETR_LIST,
        CHAIN_APPROX_SIMPLE,
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
        let moments = moments(&elem, true)?;
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
                draw_contours(
                    contour_mat,
                    &mut contours, // WHY IS THIS MUTABLE???
                    clusters[i].index as i32,
                    Scalar::all(255.),
                    1,
                    LINE_8,
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
        let testres = point_polygon_test(
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
            draw_contours(
                contour_mat,
                &mut contours,
                clusters[i].index as i32,
                color,
                1,
                LINE_8,
                &no_array(),
                i32::MAX,
                Point::default(),
            )?;
        }
        let mut cell_color = colorthresh_mat.at_pt::<Vec3b>(clusters[i].center)?.clone();
        if brightness_from_scalar(&cell_color) < 0.1 {
            cell_color = Vec3b::from_array([0, 0, 0]); // Black
            if let Some(arrow_mat) = &mut arrow_mat {
                circle(
                    arrow_mat,
                    clusters[i].center,
                    7,
                    Scalar::new(255., 215., 125., 255.),
                    1,
                    LINE_8,
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
        let mut bbox = bounding_rect(&contour)?;
        bbox.x -= 2 * (1 + DILATION_PIXELS);
        bbox.y -= 2 * (1 + DILATION_PIXELS);
        bbox.width += 4 * (1 + DILATION_PIXELS);
        bbox.height += 4 * (1 + DILATION_PIXELS);
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
        dilate_def(contour_mat, &mut a, neighbor_kernel)?;
        *contour_mat = a;
    }
    // Look for alternative
    for i in 0..cells.len() {
        for j in (i + 1)..cells.len() {
            // We know for sure they won't be the same cell as j != i
            if Cell::are_neighbors(&cells[i], &cells[j])
            || Cell::are_closeby(&cells[i], &cells[j]) {
                continue
            }
            let roi = cells[i].borrow().bbox & cells[j].borrow().bbox;
            if roi.area() <= 500 { continue }
            // This is fine because we set it to 0 later.
            unsafe {
                write_mat.create_size(roi.size(), CV_8U)?;
                dilated_mat.create_size(roi.size(), CV_8U)?;
                temp_mat.create_size(roi.size(), CV_8U)?
            };
            write_mat.set_scalar(0.into())?;
            temp_mat.set_scalar(0.into())?;
            dilated_mat.set_scalar(0.into())?;
            draw_contours(
                write_mat,
                &mut contours,
                cells[i].borrow().index as i32,
                Scalar::all(255.),
                1,
                LINE_8,
                &no_array(),
                i32::MAX,
                roi.tl() * -1
            )?;
            dilate_def(write_mat, dilated_mat, neighbor_kernel)?;
            draw_contours(
                temp_mat,
                &mut contours,
                cells[j].borrow().index as i32,
                Scalar::all(255.),
                1,
                LINE_8,
                &no_array(),
                i32::MAX,
                roi.tl() * -1
            )?;
            bitwise_and_def(dilated_mat, temp_mat, and_mat)?;
            // Only count as neighbors if enough edge overlap exists:
            // count_non_zero(and_mat) >= 2*DILATION_PIXELS ensures real shared
            // edges pass while tiny corner/diagonal contacts with few
            // pixels are ignored
            if count_non_zero(and_mat)? >= 4 * DILATION_PIXELS {
                Cell::add_neighbor(
                    &cells[i],
                &cells[j]
                );
                continue
            }
            // Also want to see if it's closeby
            dilate(
                dilated_mat,
                write_mat,
                neighbor_kernel,
                Point::new(-1, -1),
                2,
                BORDER_CONSTANT,
                border_value,
            )?;
            bitwise_and_def(write_mat, temp_mat, and_mat)?;
            if count_non_zero(and_mat)? >= 8 * DILATION_PIXELS {
                // delete when debugging is done
                if let Some(arrow_mat) = &mut arrow_mat {
                    arrowed_line(
                        arrow_mat,
                        cells[i].borrow().center,
                        cells[j].borrow().center,
                        Scalar::new(255., 0., 255., 255.),
                        1,
                        LINE_8,
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
    let mut dot_registry = Vec::<(Vec3b, [Coordinates; 2])>::new();
    println!("C {}", cells.len());
    for cell in &cells {
        if cell.borrow().is_dot {
            let mut cell = cell.borrow_mut();
            if let Some((indx, (_, arr))) = dot_registry.iter_mut().enumerate()
            .find(|(_, x)| x.0 == cell.color) {
                let location = Coordinates::from((
                    cell.center.x as usize,
                    cell.center.y as usize
                ));
                // arr[0] is guaranteed to exist
                if arr[1].is_dne() {
                    arr[1] = location;
                    cell.affiliation = indx + 1;
                } else {
                    // We somehow have more than 2 cells with one color.
                    // Uncomment when the code is perfect, right now there are
                    // some issues elsewhere
                    //return Err(opencv::Error::new(
                    //    0,
                    //    format!("{arr:?} and {location:?} all have the color {:?}", cell.color)
                    //))
                }
            } else {
                let location = Coordinates::from((
                    cell.center.x as usize,
                    cell.center.y as usize
                ));
                let aff = affiliation_displays.len() + 1;
                affiliation_displays.push(cell.color);
                cell.affiliation = aff;
                dot_registry.push((cell.color, [location, Coordinates::dne()]));
            }
        }
        let cell = cell.borrow();
        // This should be enough to detect a windmill, but if not, consider looking at the vectors
        // or directions from the nodes to the center
        'windmill: {
            // The "windmill" structure consists of straight intersecting pathways,
            // with one dominant pathway. There are some windmils with only 2 closeby,
            // and there technically could be one with 6 in the future.
            if cell.neighbors.len() != 2 || cell.closeby.len() % 2 != 0 { break 'windmill }
            let mut points = vec![Point::default(); cell.closeby.len()];
            // each closeby can only have exactly one neighbor in a windmill
            let mut rcs = Vec::with_capacity(cell.closeby.len());
            for i in 0..cell.closeby.len() {
                let rc = &cell.closeby[i];
                let closeby = rc.borrow();
                points[i] = closeby.center;
                if closeby.neighbors.len() != 1 || closeby.closeby.len() != 1 {
                    break 'windmill
                }
                drop(closeby);
                rcs.push(rc);
            };
            let mut paired = vec![usize::MAX; cell.closeby.len()];
            for i in 0..cell.closeby.len() {
                for j in (i + 1)..cell.closeby.len() {
                    if paired[i] != usize::MAX || paired[j] != usize::MAX { continue }
                    if approx_equal_slopes(points[i], points[j], cell.center) {
                        paired[i] = j;
                        paired[j] = i;
                    }
                }
            }
            if paired.contains(&usize::MAX) { break 'windmill }
            for i in 0..cell.closeby.len() {
                Cell::add_neighbor(rcs[i], rcs[paired[i]]);
            }
        }
        'bridge: {
            if cell.neighbors.len() != 4 { break 'bridge }
 
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
        println!("Stitching Took: {} seconds", (t7 - t6).as_secs_f32());
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
        let clocation = Coordinates::from((
            cell.center.x as usize,
            cell.center.y as usize
        ));
        let gcell = GraphCell::new_def(
            GraphCellHint::Empty, cell.affiliation, clocation
        );
        if cell.neighbors.len() == 0 { continue }
        graph.add_node(gcell);
        for neighbor in &cell.neighbors {
            let neighbor = neighbor.borrow();
            let nlocation = Coordinates::from((
                neighbor.center.x as usize,
                neighbor.center.y as usize
            ));
            let gneighbor = GraphCell::new_def(
                GraphCellHint::Empty, neighbor.affiliation, nlocation
            );
            graph.add_node(gneighbor);
            let res = graph.add_edge(gcell, gneighbor, GraphEdge { affiliation: 0 });
            // This is a new edge
            if res.is_none() {
                if let Some(final_mat) = &mut arrow_mat {
                    arrowed_line(
                        final_mat,
                        cell.center,
                        neighbor.center,
                        Scalar::all(255.),
                        3,
                        LINE_8,
                        0,
                        0.1
                    )?;
                }
            }
        }
    }
    let t8 = std::time::Instant::now();
    if debug_info.print_phases {
        println!("Graph Creation Took: {} seconds", (t8 - t7).as_secs_f32());
    }
    println!("NUM IN GRID: {}", graph.node_count());
    if let Some(arrow_mat) = &mut arrow_mat {
        for cell in cells.iter()
            .map(|cell| cell.borrow()).filter(|cell| cell.is_dot) {
            circle(
                arrow_mat,
                cell.center,
                7,
                Scalar::new(55., 15., 125., 255.),
                1,
                FILLED,
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