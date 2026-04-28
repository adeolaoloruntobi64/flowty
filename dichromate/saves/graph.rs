use std::io::Write;

use opencv::{
    core, highgui, imgcodecs, prelude::*, videoio, imgproc,
};
use petgraph::dot::{Config, Dot};
use rustsat_glucose::simp::Glucose;

use crate::{detectors::{CellDetector, cv::OpenCVCellDetector}, instr::Instruction, solver::ArbitraryGraphSolver};

fn main() {
	//println!("{}", core::get_build_information().unwrap());
	// C:\Users\phant\Zprograms\Rust\flowty\scripts\dataset
    let image = imgcodecs::imread("dichromate/pics/windmill.png", imgcodecs::IMREAD_COLOR).unwrap();
    //let image = imgcodecs::imread("scripts/dataset/147.png", imgcodecs::IMREAD_COLOR).unwrap();
	let mut dbginfo = DebugInfo::mostly_true();
	dbginfo.debug_save = true;
	dbginfo.print_phases = true;
	let mut detector = OpenCVCellDetector::new().unwrap();
	let start = std::time::Instant::now();
	let (graph, mats) = detector.detect_cells(&image, &dbginfo, true).unwrap();
	let end = std::time::Instant::now();
	let d = Dot::with_config(&graph, &[Config::EdgeNoLabel]);
	let mut f = std::fs::File::create("example1.dot").unwrap();
    let output = format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));
    f.write_all(&output.as_bytes()).unwrap();
	println!("Cell Detection Took: {} seconds", (end - start).as_secs_f32());
	//imgcodecs::imwrite_def("dichromate/pics2/T_gray_l.png", mats.edges_mat.as_ref().unwrap()).unwrap();
	//imgcodecs::imwrite_def("dichromate/pics2/T_gray_z.png", mats.hsv_mat.as_ref().unwrap()).unwrap();
	//imgcodecs::imwrite_def("dichromate/pics2/T.png", mats.contour_mat.as_ref().unwrap()).unwrap();
	//imgcodecs::imwrite_def("dichromate/pics2/R.png", mats.arrow_mat.as_ref().unwrap()).unwrap();
    //highgui::named_window("z", highgui::WINDOW_NORMAL).unwrap();
	//highgui::imshow("z", mats.contour_mat.as_ref().unwrap()).unwrap();
	let solver = ArbitraryGraphSolver::new(graph, Some(detector.get_affiliations().len()));
	let start = std::time::Instant::now();
    let solved = solver.solve::<Glucose>();
	println!("Solving Took: {} seconds", (std::time::Instant::now() - start).as_secs_f32());
    match solved {
        Ok(t) => {
			println!("YAY");
			let instrs = Instruction::create_vec_from_solved(&t, detector.get_affiliations().len());
			let linesiter = Instruction::to_lines_iter(instrs);
			let mut bitm = image.clone();
			for (a, b) in linesiter {
				imgproc::arrowed_line(
					&mut bitm,
					core::Point::new(a.x as i32, a.y as i32),
					core::Point::new(b.x as i32, b.y as i32),
					core::Scalar::new(255., 0., 255., 255.),
					1,
					imgproc::LINE_8,
					0,
					0.1
				).unwrap();
			}
			imgcodecs::imwrite_def("dichromate/pics2/D.png", &bitm).unwrap();
		},
        Err(e) => println!("{e:?}"),
    }
	//loop {
	//	let key = highgui::wait_key(10).unwrap();
	//	if key > 0 && key != 255 {
	//		break;
	//	}
	//}
	//println!("{}", core::get_build_information().unwrap());
}