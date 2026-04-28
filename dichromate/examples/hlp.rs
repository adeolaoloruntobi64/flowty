use std::f64::consts::PI;

use opencv::{prelude::*, imgproc::*, imgcodecs, core::*};


pub fn main() {
    let image = imgcodecs::imread("dichromate/pics/warps-3.png", imgcodecs::IMREAD_COLOR).unwrap();
    let mut gray = Mat::default();
    let mut gray2 = Mat::default();
    cvt_color_def(&image, &mut gray, COLOR_BGR2GRAY).unwrap();
    threshold(&gray, &mut gray2, 50., 255., THRESH_BINARY).unwrap();
    let mut hough: Vector<Vec4i> = Vector::new();
    hough_lines_p(
        &gray2,
        &mut hough,
        1.,
        PI / 180.,
        5,
        20.,
        25.,
    ).unwrap();
    let mut a = image.clone();
    for l in hough {
        line(
            &mut a,
            Point::new(l[0], l[1]),
            Point::new(l[2], l[3]),
            Scalar::new(0.,0.,255., 255.),
            3,
            LINE_AA,
            0
        ).unwrap();
    }
    opencv::imgcodecs::imwrite_def("dichromate/pics2/hlp.png", &a).unwrap();
}