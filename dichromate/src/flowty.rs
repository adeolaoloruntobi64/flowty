use std::{marker::PhantomData, time::{Duration, Instant}};

use enigo::InputError;
use image::{ImageBuffer, Rgba};
use opencv::{core::{CV_8UC4, Mat, Mat_AUTO_STEP, MatTraitConst, MatTraitConstManual, Point, Scalar}, imgproc};
use rustsat::solvers::{Solve, SolveStats};
use spin_sleep::SpinSleeper;
use xcap::XCapError;

use crate::{detectors::CellDetector, flowio::FlowIO, instr::Instruction, solver::{ArbitraryGraphSolver, Coordinates, SolverFailure}};

pub fn image_buffer_to_mat(img: ImageBuffer<Rgba<u8>, Vec<u8>>) -> Mat {
    let (width, height) = img.dimensions();
    let mut ret = Mat::default();
     let mat = unsafe {
         Mat::new_rows_cols_with_data_unsafe(
            height as i32,
            width as i32,
            CV_8UC4,
            img.as_ptr() as *mut _,
            Mat_AUTO_STEP
        ).unwrap()
    };
    imgproc::cvt_color(
        &mat,
        &mut ret,
        imgproc::COLOR_RGBA2BGR,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT
    ).unwrap();
    ret
}

pub struct Flowty<D: CellDetector, T: Default + Solve + SolveStats> {
    pub io: FlowIO,
    pub detector: D,
    pub phantom: PhantomData<T>
}

#[derive(Debug)]
pub enum FlowtyError {
    Enigo(InputError),
    OpenCV(opencv::Error),
    Solver(SolverFailure),
    XCap(XCapError),
}

impl<D: CellDetector, T: Default + Solve + SolveStats> Flowty<D, T> {
    pub fn new(io: FlowIO, detector: D) -> Self {
        Self { io, detector, phantom: PhantomData }
    }

    // Actual time trials dont need next_level_pos, but I have it
    // so it can simulate doing consecutive levels as time trials
    // Note next_level_pos is relative to our screenshot
    // Todo: Fix for chain levels, they do this thing where if you
    // connect and let go, there's an animation that plays and it
    // doesn't let you move aything else (chain maze and chain puzzles)
    pub fn timed_trial(
        &mut self, 
        trial_time: Duration, 
        new_board_delay: Duration, 
        update_interval: Duration,
        next_level_pos: Option<(usize, usize)>
    ) -> Result<(), FlowtyError> {
        let sleeper = SpinSleeper::default();
        let start = Instant::now();
        let cut_off = trial_time.saturating_sub(Duration::from_millis(500));
        loop {
            if Instant::now().duration_since(start) >= cut_off { break };
            self.step(update_interval)?;
            sleeper.sleep(new_board_delay / 2);
            if let Some(coords) = next_level_pos {
                self.io.execute(&[
                    Instruction::Goto(Coordinates::from(coords)),
                    Instruction::Hold,
                    Instruction::Release
                ], update_interval).map_err(FlowtyError::Enigo)?;
            }
            sleeper.sleep(new_board_delay / 2);
            let next_tick = Instant::now() + update_interval;
            let now = Instant::now();
            sleeper.sleep_until(if next_tick < now { now } else { next_tick });
        }
        Ok(())
    }

    pub fn step(&mut self, interval: Duration) -> Result<(), FlowtyError> {
        let img = self.io.capture().map_err(FlowtyError::XCap)?;
        let mat = image_buffer_to_mat(img);
        let graph = self.detector
            .detect_cells(&mat, true)
            .map_err(FlowtyError::OpenCV)?;
        let solved = ArbitraryGraphSolver::new(graph, None)
            .solve::<T>()
            .map_err(FlowtyError::Solver)?;
        let instructions = Instruction::create_vec_from_solved(
            &solved,
            self.detector.get_affiliations()
        );
        self.io.execute(&instructions, interval).map_err(FlowtyError::Enigo)?;
        Ok(())
    }

    pub fn process_pic(
        &mut self,
        pic: ImageBuffer<Rgba<u8>, Vec<u8>>
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>, FlowtyError> {
        let mat = image_buffer_to_mat(pic);
        let processed = self.process_mat(mat)?;
        let mut rgba = Mat::default();
        imgproc::cvt_color(
            &processed,
            &mut rgba,
            imgproc::COLOR_RGBA2BGR,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT
        ).unwrap();
        let buf = rgba.data_bytes().map_err(FlowtyError::OpenCV)?.to_vec();
        Ok(ImageBuffer::from_raw(rgba.cols() as u32, rgba.rows() as u32, buf).unwrap())
    }

    pub fn process_mat(&mut self, mut mat: Mat) -> Result<Mat, FlowtyError> {
        let graph = self.detector
            .detect_cells(&mat, true)
            .map_err(FlowtyError::OpenCV)?;
        let solved = ArbitraryGraphSolver::new(graph, None)
            .solve::<T>()
            .map_err(FlowtyError::Solver)?;
        let instructions = Instruction::create_vec_from_solved(
            &solved,
            self.detector.get_affiliations()
        );
        for (from, to) in Instruction::to_lines_iter(instructions) {
            imgproc::line(
                &mut mat,
                Point::new(from.x as i32, from.y as i32),
                Point::new(to.x as i32, to.y as i32),
                Scalar::all(255.),
                1,
                imgproc::LINE_8,
                0
            ).map_err(FlowtyError::OpenCV)?;
        }
        Ok(mat)
    }

}