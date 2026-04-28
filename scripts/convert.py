import os, csv
from colormath2.color_conversions import convert_color
from colormath2.color_objects import (
    sRGBColor, # RGB
    HSLColor, # HLS
    HSVColor, # HSV
    LabColor, # LAB
    LuvColor, # LUV
    LCHabColor, # Intriguing
    XYZColor, # XYZ
    # YCrCb (opencv supports but colormath2 doesn't have)
    # YUV (opencv supports but colormath2 doesn't have)
    CMYColor, # CMY (i think it's cool)
    CMYKColor # CMYK (opencv doesn't natively support, but gimp has it)
)

DATA_FILE = os.path.expanduser("~/pixel_training_dataset.csv")
DATA_FILE_MULTI = os.path.expanduser("~/pixel_training_dataset_multi.csv")

def convert_rgb(r, g, b):
    # note that by default, upscaled = false, so all values are 0 <= x <= 1
    rgb = sRGBColor(r, g, b)
    hsl: HSLColor = convert_color(rgb, HSLColor)
    hsv: HSVColor = convert_color(rgb, HSVColor)
    lab: LabColor = convert_color(rgb, LabColor)
    lch: LabColor = convert_color(rgb, LCHabColor)
    luv: LuvColor = convert_color(rgb, LuvColor)
    xyz: XYZColor = convert_color(rgb, XYZColor)
    cmy: CMYKColor = convert_color(rgb, CMYColor)
    cmyk: CMYKColor = convert_color(rgb, CMYKColor)
    
    return  rgb.get_upscaled_value_tuple(),\
            hsl.get_value_tuple(),\
            hsv.get_value_tuple(),\
            lab.get_value_tuple(),\
            lch.get_value_tuple(),\
            luv.get_value_tuple(),\
            xyz.get_value_tuple(),\
            cmy.get_value_tuple(),\
            cmyk.get_value_tuple()

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'r', newline='') as f_in, \
            open(DATA_FILE_MULTI, 'w', newline='') as f_out:
        
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        original_header = next(reader) 
        new_header = original_header + [
            "R255", "G255", "B255",
            "H_sl", "L_sl", "S_sl",
            "H_sv", "S_sv", "V_sv",
            "Lab_L", "Lab_a", "Lab_b",
            "LCh_L", "LCh_C", "LCh_h",
            "Luv_L", "Luv_u", "Luv_v",
            "XYZ_x", "XYZ_y", "XYZ_z",
            "CMY_c", "CMY_m", "CMY_y",
            "CMYK_c", "CMYK_m", "CMYK_y", "CMYK_k"
        ]
        writer.writerow(new_header)

        for row in reader:
            r, g, b = float(row[3]), float(row[4]), float(row[5])
            results = convert_rgb(r, g, b)
            flat_extra = [val for group in results for val in group]
            writer.writerow(row + flat_extra)

    print(f"Done! Saved {DATA_FILE_MULTI}")
else:
    print(f"Error: {DATA_FILE} not found!")