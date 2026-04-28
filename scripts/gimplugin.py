#!/usr/bin/env python3
import gi, os, csv, sys, struct
gi.require_version("Gimp", "3.0")
gi.require_version("Gegl", "0.4")
from gi.repository import Gimp, Gegl, GLib

DATA_FILE = os.path.expanduser("~/pixel_training_dataset.csv")

# Tried to use Babl, but guess what, Babl.fish is completely unusable in Python.
# Can't be too mad tho, at least I can get rgb. Make a separate program to
# convert into multiple spaces
def sample_pixel(drawable, x, y, label, writer):
    buffer = drawable.get_buffer()
    rect = Gegl.Rectangle.new(int(x), int(y), 1, 1)
    # Get raw bytes as floats (GIMP 3 standard)
    pixel_bytes = buffer.get(rect, 1.0, "R'G'B' float", Gegl.AbyssPolicy.NONE)
    # Unpack 3 floats (Red, Green, Blue), note that this is in %
    # i.e. 0 <= x <= 1
    r, g, b = struct.unpack('fff', pixel_bytes)
    
    writer.writerow([label, x, y, r, g, b])
    print(f"Saved {label} at {x},{y}: {r:.4f}, {g:.4f}, {b:.4f}")

    
class PixelTrainer(Gimp.PlugIn):
    def do_query_procedures(self):
        return ["path-sampler"]

    def do_create_procedure(self, name):
        proc = Gimp.ImageProcedure.new(self, name, Gimp.PDBProcType.PLUGIN, self.run, None)
        proc.set_menu_label("Collect Path Samples")
        proc.add_menu_path("<Image>/Filters")
        proc.set_image_types("*")
        return proc

    def run(self, procedure, run_mode, image, drawables, config, data):
        if not drawables:
            return procedure.new_return_values(Gimp.PDBStatusType.CALLING_ERROR, GLib.Error("No layer selected"))
        
        target_layer = drawables[0]
        count = 0
        exists = os.path.exists(DATA_FILE)

        with open(DATA_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["label", "x", "y", "R", "G", "B"])

            for label in ["good", "maybe", "bad"]:
                # Find the path by name
                vectors_list = image.get_paths()
                matches = [v for v in vectors_list if v.get_name() == label]
                
                if not matches:
                    continue
                
                # if i use .lower() above, then this has to be a for loop
                path = matches[0]
                # Get points from the path strokes
                for stroke_id in path.get_strokes():
                    # [0] is success value, and [2] is if the shape is closed
                    points = path.stroke_get_points(stroke_id)[1]
                    # Anchor points are at index 2,3 then every 6 floats
                    for i in range(2, len(points), 6):
                        x, y = int(points[i]), int(points[i+1])
                        sample_pixel(target_layer, x, y, label, writer)
                        count += 1

        Gimp.message(f"Successfully sampled {count} pixels to {DATA_FILE}")
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

Gimp.main(PixelTrainer.__gtype__, sys.argv)