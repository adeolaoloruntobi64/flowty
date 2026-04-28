use dichromate::oklch::Converter;
use opencv::{prelude::*, imgproc, imgcodecs, core::*};

fn main() {
    let image = imgcodecs::imread("dichromate/pics/chainp4.png", imgcodecs::IMREAD_COLOR).unwrap();
    let converter = Converter::new();
    let start = std::time::Instant::now();
    let lch = converter.conv_to_oklch(&image, true).unwrap();
    println!("Time took: {}", (std::time::Instant::now() - start).as_secs_f32());
    opencv::imgcodecs::imwrite_def("dichromate/pics2/ee.png", &lch).unwrap();
}