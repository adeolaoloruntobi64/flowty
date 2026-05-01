use std::time::{Duration, Instant};

use enigo::{Button, Direction, Enigo, InputResult, Mouse, Coordinate};
use image::{ImageBuffer, Rgba};
use spin_sleep::SpinSleeper;
use xcap::{Monitor, Window, XCapResult};

use crate::instr::Instruction;

pub enum Display {
    // We either capture a window
    Window(Window),
    // Or a monitor with optional dimensions (x, y, width, height)
    Monitor(Monitor, Option<(u32, u32, u32, u32)>)
}

// Struct that allows us to get "input/output", hence the "IO"
pub struct FlowIO {
    pub display: Display,
    pub engine: Enigo
}

impl FlowIO {
    pub fn new_window(window: Window, engine: Enigo) -> Self {
        Self { display: Display::Window(window), engine }
    }
    pub fn new_monitor(monitor: Monitor, engine: Enigo, region: Option<(u32, u32, u32, u32)>) -> Self {
        Self { display: Display::Monitor(monitor, region), engine }
    }

    // If we have exclusive access to the window / monitor in question
    pub fn is_exclusive(&self) -> XCapResult<bool> {
        match &self.display {
            Display::Window(window) => Ok(
                window.is_focused()? && window.current_monitor()?.is_primary()?
            ),
            Display::Monitor(monitor, _) => monitor.is_primary(),
        }
    }
    
    pub fn capture(&self) -> XCapResult<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        match &self.display {
            Display::Window(window) => window.capture_image(),
            Display::Monitor(monitor, dims) => {
                match dims {
                    &Some((x, y, width, height)) => monitor.capture_region(x, y, width, height),
                    None => monitor.capture_image()
                }
            },
        }
    }

    // I already had an implementation, but the timing was inconsistent because of the nature of std::thread::sleep
    // Wasn't sure what to do so I asked AI and it showed me this implementation using SpinSleeper. Never knew it
    // existed but W. I now use it for my time_trial implementation too
    pub fn execute(&mut self, commands: &[Instruction], interval: Duration) -> InputResult<()> {
        let sleeper = SpinSleeper::default();
        let mut next_tick = Instant::now();

        let (x_offset, y_offset) = match &self.display {
            Display::Window(_) => (0, 0),
            Display::Monitor(_, Some((x, y, _, _))) => (*x, *y),
            Display::Monitor(_, None) => (0, 0),
        };

        let (mut current_pos, commands) = if let Some(Instruction::Goto(coords)) = commands.first() {
            let start_x = coords.x as i32 + x_offset as i32;
            let start_y = coords.y as i32 + y_offset as i32;
            self.engine.move_mouse(start_x, start_y, Coordinate::Abs)?;
            ((start_x, start_y), &commands[1..commands.len()])
        } else {
            ((0, 0), commands)
        };
        for instr in commands {
            match instr {
                Instruction::Hold => {
                    self.engine.button(Button::Left, Direction::Press)?;
                    sleeper.sleep(interval);
                },
                Instruction::Release => {
                    self.engine.button(Button::Left, Direction::Release)?;
                    sleeper.sleep(interval);
                },
                Instruction::Goto(coordinates) => {
                    let target_x = coordinates.x as i32 + x_offset as i32;
                    let target_y = coordinates.y as i32 + y_offset as i32;

                    let delta_x = target_x - current_pos.0;
                    let delta_y = target_y - current_pos.1;
                    let manhattan_dist = delta_x.abs() + delta_y.abs();

                    let num_steps = (manhattan_dist / 35).max(1);

                    for i in 1..=num_steps {
                        let step_x = current_pos.0 + (delta_x * i / num_steps);
                        let step_y = current_pos.1 + (delta_y * i / num_steps);
                        
                        self.engine.move_mouse(step_x, step_y, Coordinate::Abs)?;
                        sleeper.sleep(interval);
                    }
                    current_pos = (target_x, target_y);
                },
            };

            next_tick += interval;
            let now = Instant::now();
            if next_tick < now { next_tick = now; }
            sleeper.sleep_until(next_tick);
        }
        Ok(())
    }
}