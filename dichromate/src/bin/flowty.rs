use std::time::Duration;

use enigo::{Enigo, Settings};

use dichromate::{detectors::cv::OpenCVCellDetector, flowio::FlowIO, flowty::Flowty};
use rustsat_glucose::simp::Glucose;
use xcap::Monitor;

fn main() {
	let monitors = Monitor::all().unwrap();
	let monitor = monitors.into_iter().find(|m| m.is_primary().unwrap_or(false)).unwrap();
	let engine = Enigo::new(&Settings::default()).unwrap();
	let region = Some((1734, 73, 2879 - 1734, 1677 - 73));
	let io = FlowIO::new_monitor(monitor, engine, region);
	let detector = OpenCVCellDetector::new().unwrap();
	let mut flowty = Flowty::<_, Glucose>::new(io, detector);
	// flowty.step(Duration::from_millis(20)).unwrap();
	flowty.timed_trial(
		Duration::from_mins(10),
		Duration::from_millis(1250),
		Duration::from_millis(15),
		Some((2380 - 1734, 857 - 73))
	).unwrap();
}
