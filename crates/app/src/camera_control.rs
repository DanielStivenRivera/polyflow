//! Mouse navigation for the viewport.
//!
//! Supports both Blender-style and Maya/ZBrush-style bindings:
//!
//! | Action | Blender style       | Maya style  |
//! |--------|---------------------|-------------|
//! | Orbit  | MMB drag            | Alt + LMB   |
//! | Pan    | Shift + MMB drag    | Alt + MMB   |
//! | Zoom   | Ctrl + MMB / wheel  | Alt + RMB   |

use polyflow_core::camera::OrbitCamera;
use polyflow_core::glam::Vec2;
use winit::event::MouseButton;
use winit::keyboard::ModifiersState;

const ORBIT_RADIANS_PER_PIXEL: f32 = 0.006;
const ZOOM_PER_PIXEL: f32 = 0.005;
const ZOOM_PER_WHEEL_LINE: f32 = 0.85;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DragMode {
    Orbit,
    Pan,
    Zoom,
}

#[derive(Debug, Default)]
pub struct CameraController {
    drag: Option<(DragMode, MouseButton)>,
    cursor: Option<Vec2>,
    pub modifiers: ModifiersState,
}

impl CameraController {
    /// Latest cursor position, in physical pixels.
    pub fn cursor(&self) -> Option<Vec2> {
        self.cursor
    }

    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// Camera scale factor for a wheel movement; positive `lines` zoom in.
    pub fn wheel_factor(lines: f32) -> f32 {
        ZOOM_PER_WHEEL_LINE.powf(lines)
    }

    /// Handles a button event and returns true if it started a drag. Presses
    /// over the UI never start a drag, but releases always end one.
    pub fn mouse_button(&mut self, button: MouseButton, pressed: bool, over_ui: bool) -> bool {
        if !pressed {
            if self.drag.is_some_and(|(_, b)| b == button) {
                self.drag = None;
            }
            return false;
        }
        if over_ui || self.drag.is_some() {
            return false;
        }
        let m = self.modifiers;
        let mode = match button {
            MouseButton::Left if m.alt_key() => Some(DragMode::Orbit),
            MouseButton::Middle if m.alt_key() || m.shift_key() => Some(DragMode::Pan),
            MouseButton::Middle if m.control_key() => Some(DragMode::Zoom),
            MouseButton::Middle => Some(DragMode::Orbit),
            MouseButton::Right if m.alt_key() => Some(DragMode::Zoom),
            _ => None,
        };
        self.drag = mode.map(|mode| (mode, button));
        self.drag.is_some()
    }

    /// Handles cursor motion (physical pixels). Returns true if the camera moved.
    pub fn cursor_moved(
        &mut self,
        position: Vec2,
        camera: &mut OrbitCamera,
        viewport_height: f32,
    ) -> bool {
        let previous = self.cursor.replace(position);
        let (Some((mode, _)), Some(previous)) = (self.drag, previous) else {
            return false;
        };
        let delta = position - previous;
        match mode {
            DragMode::Orbit => camera.orbit(
                -delta.x * ORBIT_RADIANS_PER_PIXEL,
                delta.y * ORBIT_RADIANS_PER_PIXEL,
            ),
            DragMode::Pan => camera.pan(delta.x, delta.y, viewport_height),
            DragMode::Zoom => camera.zoom((delta.y * ZOOM_PER_PIXEL).exp()),
        }
        delta != Vec2::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindings() {
        let mut c = CameraController::default();
        c.mouse_button(MouseButton::Middle, true, false);
        assert_eq!(c.drag, Some((DragMode::Orbit, MouseButton::Middle)));
        c.mouse_button(MouseButton::Middle, false, false);
        assert_eq!(c.drag, None);

        c.modifiers = ModifiersState::SHIFT;
        c.mouse_button(MouseButton::Middle, true, false);
        assert_eq!(c.drag, Some((DragMode::Pan, MouseButton::Middle)));
        c.mouse_button(MouseButton::Middle, false, true);
        assert_eq!(c.drag, None, "release over the UI still ends the drag");

        c.modifiers = ModifiersState::ALT;
        c.mouse_button(MouseButton::Right, true, false);
        assert_eq!(c.drag, Some((DragMode::Zoom, MouseButton::Right)));
    }

    #[test]
    fn press_over_ui_is_ignored() {
        let mut c = CameraController::default();
        c.mouse_button(MouseButton::Middle, true, true);
        assert_eq!(c.drag, None);
    }

    #[test]
    fn orbit_drag_moves_camera() {
        let mut c = CameraController::default();
        let mut cam = OrbitCamera::default();
        let before = cam;
        c.cursor_moved(Vec2::new(10.0, 10.0), &mut cam, 600.0);
        c.mouse_button(MouseButton::Middle, true, false);
        assert!(c.cursor_moved(Vec2::new(30.0, 20.0), &mut cam, 600.0));
        assert_ne!(cam.yaw, before.yaw);
        assert_ne!(cam.pitch, before.pitch);
    }
}
