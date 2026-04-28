use std::mem;

use image::{ImageBuffer, Rgba};
use opencv::core::Size;
use opencv::prelude::*;
use opencv::{highgui, videoio, Result, *};
//use rustautogui::RustAutoGui;

// This is bad btw, The mat doesn't own the datas
fn image_buffer_to_mat(img_buf: ImageBuffer<Rgba<u8>, Vec<u8>>) -> Mat {
    let width = img_buf.width() as i32;
    let height = img_buf.height() as i32;

    // Get the raw pixel data as a Vec<u8>
    let pixels = img_buf.into_raw();

    // Create a Mat from the buffer.
    let mat = unsafe {
        Mat::new_rows_cols_with_data_unsafe(
            height,
            width,
            opencv::core::CV_8UC4,
            pixels.as_ptr() as *mut _,
            opencv::core::Mat_AUTO_STEP
        ).unwrap()
    };
    mem::forget(pixels);
    mat
}

fn main() {
    let window = "video capture";
	//highgui::named_window(window, highgui::WINDOW_AUTOSIZE).unwrap();
    //let mut gui = RustAutoGui::new(false).unwrap();
	//while highgui::get_window_property(window, highgui::WND_PROP_VISIBLE).unwrap() >= 1.0 {
    //    gui.screen.capture_screen();
    //    let img = gui.screen.convert_bitmap_to_rgba().unwrap();
	//	let frame = image_buffer_to_mat(img);
    //    if frame.size().unwrap().width > 0 {
	//		highgui::imshow(window, &frame).unwrap();
	//	}
	//	let key = highgui::wait_key(10).unwrap();
	//	if key == 27 {
	//		break;
	//	}
	//}
}