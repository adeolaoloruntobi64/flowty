use enigo::{Button, Direction, Enigo, InputResult, Mouse, Coordinate, Settings};

fn main() {
    let mut enigo = Enigo::new(&Settings::default()).unwrap();
    enigo.move_mouse(200, 200, Coordinate::Abs).unwrap();
    enigo.button(Button::Left, Direction::Click).unwrap();
    enigo.move_mouse(400, 400, Coordinate::Abs).unwrap();
    enigo.move_mouse(600, 600, Coordinate::Abs).unwrap();
    enigo.move_mouse(800, 600, Coordinate::Abs).unwrap();
    enigo.move_mouse(1000, 600, Coordinate::Abs).unwrap();
}