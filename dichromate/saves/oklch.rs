use std::f32::consts::PI;

use opencv::core::{CV_32FC3, Mat, MatTrait, MatTraitConst};
use rayon::prelude::*;

const BLOCK_ROWS: usize = 32;
pub struct Converter {
    linear_lut: [f32; 256]
}

impl Converter {

    pub fn new() -> Self {
        Self { linear_lut: Self::build_srgb_to_linear_lut() }
    }

    fn build_srgb_to_linear_lut() -> [f32; 256] {
        let mut lut = [0.0f32; 256];

        for i in 0..256 {
            let c = i as f32 / 255.0;

            lut[i] = if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            };
        }

        lut
    }

    #[inline(always)]
    // https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
    fn pixel_to_oklch(&self, r: u8, g: u8, b: u8) -> (f32, f32, f32) {
        // sRGB → linear
        let r = self.linear_lut[r as usize];
        let g = self.linear_lut[g as usize];
        let b = self.linear_lut[b as usize];

        // linear RGB → LMS
        let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
        let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
        let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

        let l = l.cbrt();
        let m = m.cbrt();
        let s = s.cbrt();

        // OKLab
        #[allow(non_snake_case)]
        let L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
        let a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
        let b = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;

        // OKLCH
        #[allow(non_snake_case)]
        let C = (a.powi(2) + b.powi(2)).sqrt();
        let h = b.atan2(a);

        (L, C, h)
    }

    #[inline(always)]
    fn arr_to_oklch_f32(&self, bgr: bool, width: usize, height: usize, pixels: &[u8], out: &mut [f32]) {
        assert!(pixels.len() % 3 == 0);
        assert!(out.len() == pixels.len());

        println!("{:?} {:?}", &out[0..3], &pixels[0..3]);
        out.par_chunks_mut(width * 3 * BLOCK_ROWS)
            .enumerate()
            .for_each(|(block_idx, out_block)| {
                let y0 = block_idx * BLOCK_ROWS;
                let y1 = (y0 + BLOCK_ROWS).min(height);

                for (row, out_row) in (y0..y1).zip(out_block.chunks_mut(width * 3)) {
                    let in_row = row * width * 3;
                    let in_slice = &pixels[in_row..in_row + width * 3];

                    for x in 0..width {
                        let i = x * 3;
                        let (r, g, b) = if bgr {
                            (in_slice[i + 2], in_slice[i + 1], in_slice[i])
                        } else {
                            (in_slice[i], in_slice[i + 1], in_slice[i + 2])
                        };
                        let (l, c, h) = self.pixel_to_oklch(r, g, b);
                        out_row[i] = l * 255.;
                        out_row[i + 1] = c * 255.;
                        out_row[i + 2] = (h * 180. / PI).rem_euclid(360.0) * 255. / 360.;
                    }
                }
            });
        println!("{:?}", &out[0..3]);
    }

    pub fn conv_to_oklch(&self, mat: &Mat, bgr: bool) -> opencv::Result<Mat> {
        assert!(mat.is_continuous());
        unsafe {
            // r g b / b g r  ---> l c h, 3 values per pixel
            let size = 3 * (mat.rows() * mat.cols()) as usize;
            // Is continuous by definition
            let mut new_mat = Mat::new_rows_cols(
                mat.rows(),
                mat.cols(),
                CV_32FC3
            )?;
            let pixels = std::slice::from_raw_parts(
                mat.data(),
                size
            );
            let out = std::slice::from_raw_parts_mut(
                new_mat.data_mut() as *mut f32,
                size
            );
            self.arr_to_oklch_f32(bgr, mat.cols() as usize, mat.rows() as usize, pixels, out);
            Ok(new_mat)
        }
    }

}