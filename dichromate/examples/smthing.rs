use opencv::{prelude::*, imgproc, imgcodecs, core::*};

fn main() {
    let image = imgcodecs::imread("dichromate/pics/y.png", imgcodecs::IMREAD_COLOR).unwrap();
    let mut imgf = Mat::default();
    image.convert_to_def(&mut imgf, CV_32F).unwrap();
    let [rows, cols] = [imgf.rows(), imgf.cols()];
    let samples = imgf.reshape(1, rows * cols).unwrap();
    let mut labels = Mat::default();
    let mut centers = Mat::default();
    kmeans(
        &samples,
        20,
        &mut labels,
        TermCriteria::new(
            TermCriteria_EPS +
            TermCriteria_COUNT,
            1,
            0.
        ).unwrap(),
        1,
        KMEANS_PP_CENTERS,
        &mut centers
    ).unwrap();
    let mut res = Mat::zeros(rows, cols, CV_32FC3).unwrap().to_mat().unwrap();
    for y in 0..rows {
        for x in 0..cols {

            let idx = (y * cols + x) as i32;

            let label = *labels.at::<i32>(idx).unwrap();

            let b = *centers.at_2d::<f32>(label, 0).unwrap();
            let g = *centers.at_2d::<f32>(label, 1).unwrap();
            let r = *centers.at_2d::<f32>(label, 2).unwrap();

            let pixel = res.at_2d_mut::<Vec3f>(y, x).unwrap();
            pixel[0] = b;
            pixel[1] = g;
            pixel[2] = r;
        }
    }
    opencv::imgcodecs::imwrite_def("dichromate/pics2/ee.png", &res).unwrap();
}