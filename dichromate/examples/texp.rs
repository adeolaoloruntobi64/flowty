use std::fs;
use dichromate::detectors::{CellDetector, cv::OpenCVCellDetector};
use opencv::{
    core, highgui, imgcodecs, imgproc::COLOR_BGR2HSV_FULL, prelude::*, videoio
};

fn main() {
	//println!("{}", core::get_build_information().unwrap());
    //let mut dbginfo = DebugInfo::mostly_true();
    //dbginfo.edges_mat = false;
	let mut detector = OpenCVCellDetector::new().unwrap();
    let start = std::time::Instant::now();
    let files = fs::read_dir("dichromate/pics").unwrap();
    let mut len = 0;
    for rentry in files {
        let Ok(entry) = rentry else { continue };
        let path = entry.path();
        if !path.is_file() { continue };
        let Some(osname) = path.file_name() else { continue };
        let Some(name)  = osname.to_str() else { continue };
        println!("Testing: {name}");
        let src = format!("dichromate/pics/{name}");
        let dst = format!("dichromate/pics3/board_{name}");
        let image = imgcodecs::imread(&src, imgcodecs::IMREAD_COLOR).unwrap();
        //let mat = detector.detect_cells2(&image, &dbginfo, true).unwrap();
        //imgcodecs::imwrite_def(&dst,& mat).unwrap();
        //len += 1;
    }
    let end = std::time::Instant::now();
    let nsecs = (end - start).as_secs_f32();
    let frac = nsecs / len as f32;
    println!("Testing {len} items took: {nsecs} seconds, or {frac} seconds per image on average");
}