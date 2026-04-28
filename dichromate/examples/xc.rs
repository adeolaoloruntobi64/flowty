use image::{ImageBuffer, Rgba};
use xcap::Window;
use opencv::{boxed_ref::BoxedRef, core::*, highgui, imgproc, prelude::*};


// These two both have access to the same data, but one owns it, while the other
// just references it
pub struct GroupedPair {
    img: ImageBuffer<Rgba<u8>, Vec<u8>>,
    mat: Mat
}

pub fn image_buffer_to_mat(img: ImageBuffer<Rgba<u8>, Vec<u8>>) -> GroupedPair {
    let (width, height) = img.dimensions();
     let mat = unsafe {
         Mat::new_rows_cols_with_data_unsafe(
            height as i32,
            width as i32,
            CV_8UC4,
            img.as_ptr() as *mut _,
            Mat_AUTO_STEP
        ).unwrap()
    };
    GroupedPair { img, mat }
}

fn main() {
    let window_name = "video capture";
    let name = "Visual Studio Code";
    let windows = Window::all().unwrap();
    let window = windows.iter().find(
        |window| window.app_name().unwrap().contains(name)
    ).unwrap();
	highgui::named_window(window_name, highgui::WINDOW_GUI_NORMAL).unwrap();
	while highgui::get_window_property(window_name, highgui::WND_PROP_VISIBLE).unwrap() >= 1.0 {
        let img = window.capture_image().unwrap();
        let pair = image_buffer_to_mat(img);
        let mut f2 = Mat::default();
        imgproc::cvt_color(
            &pair.mat, 
            &mut f2,
            imgproc::COLOR_BGR2RGB,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT
        ).unwrap();
        if pair.mat.size().unwrap().width > 0 {
			highgui::imshow(window_name, &f2).unwrap();
		}
		let key = highgui::wait_key(10).unwrap();
		if key == 27 {
			break;
		}
	}
}