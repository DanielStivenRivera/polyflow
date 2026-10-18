//! Orbit camera used by the viewport.
//!
//! Conventions: right-handed, +Y up. Projection matrices target Vulkan clip
//! space (Y down, depth in `[0, 1]`) with reverse-Z and an infinite far plane,
//! which gives near-uniform depth precision for huge scenes.

use glam::{Mat4, Vec3};

use crate::bounds::Aabb;

/// Keeps the camera from flipping over the poles.
const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 1e-3;
const MIN_DISTANCE: f32 = 1e-4;

/// A camera orbiting around a target point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    /// Point the camera orbits around and looks at.
    pub target: Vec3,
    /// Distance from the target to the eye.
    pub distance: f32,
    /// Rotation around the world Y axis, in radians.
    pub yaw: f32,
    /// Elevation above the XZ plane, in radians.
    pub pitch: f32,
    /// Vertical field of view, in radians.
    pub fov_y: f32,
    /// Near clip plane distance. The far plane is at infinity.
    pub near: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 5.0,
            yaw: 0.0,
            pitch: 0.3,
            fov_y: 45f32.to_radians(),
            near: 0.01,
        }
    }
}

impl OrbitCamera {
    /// Unit vector from the target towards the eye.
    pub fn back(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, sp, cy * cp)
    }

    /// Eye position in world space.
    pub fn eye(&self) -> Vec3 {
        self.target + self.back() * self.distance
    }

    /// World-to-view matrix.
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    /// View-to-clip matrix for Vulkan: reverse-Z, infinite far plane, Y down.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        let mut proj = Mat4::perspective_infinite_reverse_rh(self.fov_y, aspect, self.near);
        proj.y_axis.y = -proj.y_axis.y;
        proj
    }

    /// World-to-clip matrix.
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }

    /// Rotates around the target by the given yaw and pitch deltas, in radians.
    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw = (self.yaw + delta_yaw).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch + delta_pitch).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Moves the target in the view plane so that the surface at the target
    /// depth follows the cursor. Deltas are in pixels, Y down.
    pub fn pan(&mut self, dx: f32, dy: f32, viewport_height: f32) {
        let world_per_pixel = self.world_per_pixel(viewport_height);
        let back = self.back();
        let right = Vec3::Y.cross(back).normalize();
        let up = back.cross(right);
        self.target += (-right * dx + up * dy) * world_per_pixel;
    }

    /// Multiplies the distance to the target by `factor` (< 1 moves closer).
    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).max(MIN_DISTANCE);
        self.near = self.near.min(self.distance * 0.1);
    }

    /// Centers the camera on `bounds` and backs off until it fits a view with
    /// the given aspect ratio (width / height).
    pub fn frame(&mut self, bounds: &Aabb, aspect: f32) {
        let radius = bounds.radius().max(MIN_DISTANCE);
        let half_fov_y = self.fov_y * 0.5;
        let half_fov_x = (half_fov_y.tan() * aspect).atan();
        self.target = bounds.center();
        self.distance = radius / half_fov_y.min(half_fov_x).sin();
        self.near = radius * 1e-3;
    }

    /// Size in world units of one pixel at the target depth.
    pub fn world_per_pixel(&self, viewport_height: f32) -> f32 {
        2.0 * self.distance * (self.fov_y * 0.5).tan() / viewport_height.max(1.0)
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec4Swizzles;

    use super::*;

    fn to_ndc(m: Mat4, p: Vec3) -> Vec3 {
        let clip = m * p.extend(1.0);
        clip.xyz() / clip.w
    }

    #[test]
    fn target_projects_to_screen_center() {
        let cam = OrbitCamera {
            target: Vec3::new(1.0, 2.0, 3.0),
            yaw: 0.7,
            pitch: -0.4,
            ..Default::default()
        };
        let ndc = to_ndc(cam.view_projection(16.0 / 9.0), cam.target);
        assert!(ndc.x.abs() < 1e-5 && ndc.y.abs() < 1e-5, "{ndc}");
    }

    #[test]
    fn reverse_z_and_vulkan_y_down() {
        let cam = OrbitCamera {
            pitch: 0.0,
            ..Default::default()
        };
        let vp = cam.view_projection(1.0);
        let near = to_ndc(vp, cam.target + cam.back() * 1.0);
        let far = to_ndc(vp, cam.target - cam.back() * 100.0);
        assert!(near.z > far.z, "closer points must have larger depth");
        assert!((0.0..=1.0).contains(&near.z) && (0.0..=1.0).contains(&far.z));
        let above = to_ndc(vp, cam.target + Vec3::Y * 0.5);
        assert!(above.y < 0.0, "world up must map to screen up (negative Y)");
    }

    #[test]
    fn pitch_is_clamped() {
        let mut cam = OrbitCamera::default();
        cam.orbit(0.0, 10.0);
        assert!(cam.pitch < std::f32::consts::FRAC_PI_2);
        cam.orbit(0.0, -20.0);
        assert!(cam.pitch > -std::f32::consts::FRAC_PI_2);
    }

    #[test]
    fn pan_tracks_cursor_at_target_depth() {
        let mut cam = OrbitCamera {
            yaw: 1.1,
            pitch: 0.5,
            ..Default::default()
        };
        let height = 800.0;
        let point = cam.target;
        let before = to_ndc(cam.view_projection(1.0), point);
        cam.pan(40.0, -25.0, height);
        let after = to_ndc(cam.view_projection(1.0), point);
        // NDC spans 2 units over `height` pixels.
        let moved_px = (after - before).truncate() * height / 2.0;
        assert!((moved_px.x - 40.0).abs() < 1e-2, "{moved_px}");
        assert!((moved_px.y + 25.0).abs() < 1e-2, "{moved_px}");
    }

    #[test]
    fn frame_fits_bounds() {
        let mut cam = OrbitCamera::default();
        let bounds = Aabb {
            min: Vec3::splat(-10.0),
            max: Vec3::new(30.0, 10.0, 10.0),
        };
        for aspect in [0.4, 1.0, 2.5] {
            cam.frame(&bounds, aspect);
            assert_eq!(cam.target, bounds.center());
            let vp = cam.view_projection(aspect);
            for corner in bounds.corners() {
                let ndc = to_ndc(vp, corner);
                assert!(
                    ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0,
                    "aspect {aspect}: {corner} -> {ndc}"
                );
            }
        }
    }
}
