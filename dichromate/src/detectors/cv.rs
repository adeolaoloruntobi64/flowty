use std::{cell::RefCell, collections::VecDeque, f64::consts::PI, rc::Rc};
use itertools::Itertools;
use opencv::{
    core::*, *, imgproc::*
};
use crate::{detectors::{CellDetector, SupportedFeatures}, solver::{Coordinates, GraphCell, GraphCellHint, GraphEdge}};
use petgraph::prelude::UnGraphMap;
use rayon::prelude::*;

const DILATION_PIXELS: i32 = 4;

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
    pub area: f64,
    pub is_dot: bool,
    pub is_phantom: bool,
    pub is_fake: bool,
    // Do note than a cell in one cannot be in the other
    pub neighbors: Vec<Rc<RefCell<Cell>>>,
    pub closeby: Vec<Rc<RefCell<Cell>>>,
    pub bbox: Rect,
    pub index: usize,
    pub vec_index: usize,
    pub affiliation: usize
}

impl Cell {
    pub fn new_def(
        color: Vec3b,
        center: Point,
        area: f64,
        bbox: Rect,
        index: usize,
        vec_index: usize,
        is_phantom: bool,
    ) -> Self {
        Self {
            color,
            center,
            area,
            neighbors: Vec::new(),
            closeby: Vec::new(),
            is_dot: color != Vec3b::from_array([0, 0, 0]),
            is_phantom,
            is_fake: false,
            bbox,
            index,
            vec_index,
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

fn distance_between(a: &Point, b: &Point) -> f64 {
    return (
        (a.x - b.x).pow(2) as f64 + 
        (a.y - b.y).pow(2) as f64
    ).sqrt()
}

fn approx_equal_slopes(p1: Point, p2: Point, center: Point) -> bool {
    let dx1 = p1.x - center.x;
    let dy1 = p1.y - center.y;
    let dx2 = p2.x - center.x;
    let dy2 = p2.y - center.y;

    // Check if vectors are parallel: dx1*dy2 == dx2*dy1
    let cross = (dx1*dy2 - dx2*dy1) as f64;
    //let max_mag = ((dx1.abs().max(dy1.abs())) * (dx2.abs().max(dy2.abs()))) as f64;
    cross.abs() <= 100. //1e-12f64.max(1e-9f64 * max_mag)
}

pub struct OpenCVCellDetector {
    hsv_mat: Mat,
    write_mat: Mat,
    temp_mat: Mat,
    blob_mat: Mat,
    thresh_mat: Mat,
    colorthresh_mat: Mat,
    hsvthresh_mat: Mat,
    and_mat: Mat,
    dilated_mat: Mat,
    ver_kernel: Mat,
    hor_kernel: Mat,
    dil_kernel: Mat,
    neighbor_kernel: Mat,
    border_value: Scalar,
    affiliation_displays: Vec<Vec3b>,
}

impl OpenCVCellDetector {
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
            ver_kernel: get_structuring_element(
                MORPH_RECT,
                Size::new(25, 1),
                Point::new(-1, -1)
            )?,
            hor_kernel: get_structuring_element(
                MORPH_RECT,
                Size::new(1, 25),
                Point::new(-1, -1)
            )?,
            dil_kernel: get_structuring_element(
                MORPH_RECT,
                Size::new(5, 5),
                Point::new(-1, -1)
            )?,
            neighbor_kernel: get_structuring_element(
                MORPH_ELLIPSE,
                Size::new(2 * DILATION_PIXELS + 1, 2 * DILATION_PIXELS + 1),
                Point::new(-1, -1)
            )?,
            // Default border value
            border_value: morphology_default_border_value()?,
            // Our internal reference for terminal nodes
            affiliation_displays: Vec::new(),
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
}

impl CellDetector for OpenCVCellDetector {
    const SUPPORTED_FEATURES: SupportedFeatures = SupportedFeatures::new()
        .with_rectangle()
        .with_shapes()
        .with_chains();

    fn get_affiliations(&self) -> &Vec<Vec3b> {
        &self.affiliation_displays
    }

    fn detect_cells(
        &mut self,
        bit_mat: &Mat,
        bgr: bool
    ) -> opencv::error::Result<UnGraphMap<GraphCell, GraphEdge>> {
        self.clean()?;
        let code = if bgr { COLOR_BGR2HSV_FULL } else { COLOR_RGB2HSV_FULL };
        cvt_color(
            &bit_mat, 
            &mut self.temp_mat,
            code,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT
        )?;
        // #2: Find pixels with low V values (store in write)
        // Remove small bg noise and unwanted segments
        in_range(
            &self.temp_mat,
            &Vec3b::from_array([0, 0, 0]),
            &Vec3b::from_array([255, 255, 56]),
            &mut self.thresh_mat,
        )?;
        // #3: Get (most) blobs
        in_range(
            &self.temp_mat,
            &Vec3b::from([0, 225, 0]),
            &Vec3b::from([255, 255, 90]),
            &mut self.blob_mat
        )?;
        // store the union in write
        bitwise_or_def(&self.thresh_mat, &self.blob_mat, &mut self.write_mat)?;
        // Remember we want to get rid of what passed (white), so invert
        bitwise_not_def(&self.write_mat, &mut self.thresh_mat)?;
        // hsv mat with all the noise gone
        self.temp_mat.copy_to_masked(&mut self.hsv_mat, &self.thresh_mat)?;
        // Mask of non zero pixels
        in_range(
            &self.hsv_mat,
            &Vec3b::from_array([0, 0, 1]),
            &Vec3b::from_array([255, 255, 255]),
            &mut self.temp_mat
        )?;
        // here we patch perpendicular lines that don't connect because of
        // issues. This allows us to support more shapes puzzles now. I assume
        // noone else will see this, so to me, this exists because of numbers
        // 266 and 364 in the dataset.
        morphology_ex(
            &self.temp_mat,
            &mut self.write_mat,
            MORPH_OPEN,
            &self.ver_kernel,
            Point::new(-1, -1),
            4,
            BORDER_CONSTANT,
            self.border_value,
        )?;
        morphology_ex(
            &self.temp_mat,
            &mut self.blob_mat,
            MORPH_OPEN,
            &self.hor_kernel,
            Point::new(-1, -1),
            4,
            BORDER_CONSTANT,
            self.border_value,
        )?;
        bitwise_or_def(&self.write_mat, &self.blob_mat, &mut self.hsvthresh_mat)?;
        morphology_ex(
            &self.hsvthresh_mat,
            &mut self.blob_mat,
            MORPH_CLOSE,
            &self.dil_kernel,
            Point::new(-1, -1),
            2,
            BORDER_CONSTANT,
            self.border_value,
        )?;
        bitwise_or_def(&self.temp_mat, &self.blob_mat, &mut self.write_mat)?;
        let mut contours = Vector::<Vector::<Point>>::new();
        // get the contours of the processed image
        find_contours(
            &self.write_mat,
            &mut contours,
            RETR_LIST,
            CHAIN_APPROX_SIMPLE,
            Point::new(-1, -1)
        )?;
        let mut clusters = Vec::<Cluster>::with_capacity(contours.len());
        for (i, elem) in contours.iter().enumerate() {
            // (https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html#:~:text=From%20this%20moments,M%5B%27m00%27%5D.)
            let moments = moments(&elem, true)?;
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
                is_bad: area < 800.0 || area > 60000.0
            };
            clusters.push(cluster);
        }    
        let mut cells: Vec<Rc<RefCell<Cell>>> = Vec::new();
        let imgbound = Rect::new(
            0,
            0,
            self.temp_mat.cols(),
            self.temp_mat.rows()
        );
        clusters.sort_by(|a, b| b.area.total_cmp(&a.area));
        for i in 0..clusters.len() {
            // Make sure contour is not too big or too small
            if clusters[i].is_bad { continue }
            let contour = contours.get(clusters[i].index)?;
            if 'a: {
                for cell2 in &cells {
                    let mut c2 = cell2.borrow_mut();
                    // A dot inside a cell will have approx the same center as the cell
                    // Check if 2 centers are approx the same. If they are,
                    // then delete the most recent one, which should be the circle
                    // since we sort contours by size descending

                    // Want to keep
                    // Cell vs dot: Keep cell
                    // Small vs big: keep small
                    // We know the biggest ones are inserted first
                    // If what exists is significantly bigger, remove it and process this one
                    // else, don't consider our current
                    if distance_between(&clusters[i].center, &c2.center) < 12. {
                        if c2.area >= 2.7 * clusters[i].area && contour.len() < 10  { 
                            c2.is_fake = true;
                        } else {
                            break 'a true;
                        }
                    }
                }
                false
            } { continue }
            let mut cell_color = bit_mat.at_pt::<Vec3b>(clusters[i].center)?.clone();
            let hsv_color = self.hsv_mat.at_pt::<Vec3b>(clusters[i].center)?.clone();
            if hsv_color[2] < 127 {
                cell_color = Vec3b::from_array([0, 0, 0]);
            }
            let mut bbox = bounding_rect(&contour)?;
            bbox.x -= 2 * (1 + DILATION_PIXELS);
            bbox.y -= 2 * (1 + DILATION_PIXELS);
            bbox.width += 4 * (1 + DILATION_PIXELS);
            bbox.height += 4 * (1 + DILATION_PIXELS);
            bbox &= imgbound;
            cells.push(Rc::new(RefCell::new(Cell::new_def(
                cell_color,
                clusters[i].center,
                clusters[i].area,
                bbox,
                clusters[i].index,
                cells.len(),
                false
            ))));
        }
        for i in 0..cells.len() {
            for j in (i + 1)..cells.len() {
                // We know for sure they won't be the same cell as j != i
                if Cell::are_neighbors(&cells[i], &cells[j])
                || Cell::are_closeby(&cells[i], &cells[j])
                || cells[i].borrow().is_fake || cells[j].borrow().is_fake
                { continue }
                let roi = cells[i].borrow().bbox & cells[j].borrow().bbox;
                if roi.area() <= 500 { continue }
                // This is fine because we set it to 0 later.
                unsafe {
                    self.write_mat.create_size(roi.size(), CV_8U)?;
                    self.dilated_mat.create_size(roi.size(), CV_8U)?;
                    self.temp_mat.create_size(roi.size(), CV_8U)?
                };
                self.write_mat.set_scalar(0.into())?;
                self.temp_mat.set_scalar(0.into())?;
                self.dilated_mat.set_scalar(0.into())?;
                draw_contours(
                    &mut self.write_mat,
                    &mut contours,
                    cells[i].borrow().index as i32,
                    Scalar::all(255.),
                    1,
                    LINE_8,
                    &no_array(),
                    i32::MAX,
                    roi.tl() * -1
                )?;
                dilate_def(&self.write_mat, &mut self.dilated_mat, &mut self.neighbor_kernel)?;
                draw_contours(
                    &mut self.temp_mat,
                    &mut contours,
                    cells[j].borrow().index as i32,
                    Scalar::all(255.),
                    1,
                    LINE_8,
                    &no_array(),
                    i32::MAX,
                    roi.tl() * -1
                )?;
                bitwise_and_def(&self.dilated_mat, &self.temp_mat, &mut self.and_mat)?;
                // Only count as neighbors if enough edge overlap exists
                if count_non_zero(&self.and_mat)? >= 4 * DILATION_PIXELS {
                    Cell::add_neighbor(
                        &cells[i],
                    &cells[j]
                    );
                    continue
                }
                // Also want to see if it's closeby
                dilate(
                    &self.dilated_mat,
                    &mut self.write_mat,
                    &self.neighbor_kernel,
                    Point::new(-1, -1),
                    2,
                    BORDER_CONSTANT,
                    self.border_value,
                )?;
                bitwise_and_def(&self.write_mat, &self.temp_mat, &mut self.and_mat)?;
                if count_non_zero(&self.and_mat)? >= 6 * DILATION_PIXELS {
                    Cell::add_closeby(
                        &cells[i],
                    &cells[j]
                    );
                }
            }
        }
        // Also need to get rid of cells not part of the big cluster
        for cell in cells.iter().filter(|c| !c.borrow().is_fake) {
            let cell = cell.borrow();
            // This should be enough to detect a windmill, but if not, consider looking at the vectors
            // or directions from the nodes to the center
            'windmill: {
                // The "windmill" structure consists of straight intersecting pathways,
                // with one dominant pathway. There are some windmils with only 2 closeby,
                // and there technically could be one with 6 in the future.
                if cell.neighbors.len() != 2 || cell.closeby.len() % 2 != 0 { break 'windmill }
                let mut points = Vec::new();
                // each closeby can only have exactly one neighbor in a windmill
                let mut rcs = Vec::with_capacity(cell.closeby.len());
                for i in 0..cell.closeby.len() {
                    let rc = &cell.closeby[i];
                    let closeby = rc.borrow();
                    if closeby.neighbors.len() != 1 || closeby.neighbors[0].borrow().is_fake || closeby.is_dot { continue }
                    points.push(closeby.center);
                    drop(closeby);
                    rcs.push(rc);
                };
                let mut paired = vec![usize::MAX; points.len()];
                for i in 0..points.len() {
                    for j in (i + 1)..points.len() {
                        if paired[i] != usize::MAX || paired[j] != usize::MAX { continue }
                        if approx_equal_slopes(points[i], points[j], cell.center) {
                            paired[i] = j;
                            paired[j] = i;
                        }
                    }
                }
                if paired.contains(&usize::MAX) { break 'windmill }
                for i in 0..points.len() {
                    Cell::add_neighbor(rcs[i], rcs[paired[i]]);
                }
            }
        }

        let mut dot_registry = Vec::<(Vec3b, [(usize, Coordinates); 2])>::new();
        let mut queue = (0..cells.len()).collect::<VecDeque<_>>();
        loop {
            // worklist fixpoint
            while let Some(idx) = queue.pop_front() {
                let mut this = cells[idx].borrow_mut();
                if this.is_fake { continue; }
                let valid_neighbor_count = this.neighbors.iter()
                    .filter(|n| !n.borrow().is_fake)
                    .count();
                let should_be_fake = (this.is_dot && valid_neighbor_count < 1)
                    || (!this.is_dot && valid_neighbor_count < 2);
                if should_be_fake {
                    this.is_fake = true;
                    drop(this);
                    // only recheck neighbors of the newly faked cell
                    for neighbor in cells[idx].borrow().neighbors.iter() {
                        let n = neighbor.borrow();
                        if !n.is_fake {
                            queue.push_back(n.vec_index);
                        }
                    }
                }
            }

            // rebuild registry from surviving dots
            dot_registry.clear();
            for (idx, cell) in cells.iter().enumerate() {
                let cell = cell.borrow();
                if cell.is_fake || !cell.is_dot { continue; }
                let location = Coordinates::from((
                    cell.center.x as usize,
                    cell.center.y as usize,
                ));
                if let Some((_, arr)) = dot_registry.iter_mut()
                    .find(|x| x.0 == cell.color)
                {
                    if arr[1].1.is_dne() {
                        arr[1] = (idx, location);
                    } else {
                        return Err(opencv::Error::new(
                            0,
                            format!("{arr:?} and {location:?} all have the color {:?}", cell.color)
                        ));
                    }
                } else {
                    dot_registry.push((cell.color, [(idx, location), (usize::MAX, Coordinates::dne())]));
                }
            }

            // mark unpaired dots fake and re-run fixpoint if any were found
            let mut pair_removed = false;
            dot_registry.retain(|(_, arr)| {
                if arr[1].1.is_dne() {
                    let mut cell = cells[arr[0].0].borrow_mut();
                    cell.is_fake = true;
                    // seed neighbors of this newly faked dot back into the queue
                    for neighbor in cell.neighbors.iter() {
                        let n = neighbor.borrow();
                        if !n.is_fake {
                            queue.push_back(n.vec_index);
                        }
                    }
                    pair_removed = true;
                    false
                } else {
                    true
                }
            });

            if !pair_removed { break; }
        }

        self.affiliation_displays.clear();
        for (i, (color, arr)) in dot_registry.iter().enumerate() {
            self.affiliation_displays.push(*color);
            cells[arr[0].0].borrow_mut().affiliation = i + 1;
            cells[arr[1].0].borrow_mut().affiliation = i + 1;
        }

        // 4 times is probably a good estimate
        let mut graph = UnGraphMap::<GraphCell, GraphEdge>::with_capacity(
            cells.len(),
            4 * cells.len()
        );
        
        for cell in &cells {
            let cell = cell.borrow();
            if cell.is_fake { continue }
            let clocation = Coordinates::from((
                cell.center.x as usize,
                cell.center.y as usize
            ));
            let gcell = GraphCell::new_def(
                GraphCellHint::Empty, cell.affiliation, clocation
            );
            graph.add_node(gcell);
            for neighbor in &cell.neighbors {
                let neighbor = neighbor.borrow();
                if neighbor.is_fake { continue }
                let nlocation = Coordinates::from((
                    neighbor.center.x as usize,
                    neighbor.center.y as usize
                ));
                let gneighbor = GraphCell::new_def(
                    GraphCellHint::Empty, neighbor.affiliation, nlocation
                );
                graph.add_node(gneighbor);
                graph.add_edge(gcell, gneighbor, GraphEdge { affiliation: 0 });
            }
        }
        return Ok(graph);
    }
}