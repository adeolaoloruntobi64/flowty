use std::time::Duration;

use enigo::{Enigo, Settings};
use opencv::imgcodecs;
use dichromate::{detectors::cv::OpenCVCellDetector, flowio::FlowIO, flowty::Flowty};
use rustsat_glucose::simp::Glucose;
use xcap::Monitor;

fn main() {
    let img = imgcodecs::imread("boards/og/bridge-1.png", imgcodecs::IMREAD_COLOR).unwrap();
	let monitors = Monitor::all().unwrap();
	let monitor = monitors.into_iter().find(|m| m.is_primary().unwrap_or(false)).unwrap();
	let engine = Enigo::new(&Settings::default()).unwrap();
	let region = Some((1734, 73, 2879 - 1734, 1677 - 73));
	let io = FlowIO::new_monitor(monitor, engine, region);
	let detector = OpenCVCellDetector::new().unwrap();
	let mut flowty = Flowty::<_, Glucose>::new(io, detector);
	let res = flowty.process_mat(img).unwrap();
	imgcodecs::imwrite_def("boards/rest.png", &res).unwrap();

}
