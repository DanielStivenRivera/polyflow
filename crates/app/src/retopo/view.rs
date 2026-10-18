//! Screen-space helpers: projecting points and casting rays from the cursor.

use polyflow_core::camera::OrbitCamera;
use polyflow_core::glam::{Mat4, Vec2, Vec3};
use polyflow_render::Viewport;

/// The camera and viewport of the current frame.
#[derive(Debug, Clone, Copy)]
pub struct ViewInfo {
    pub camera: OrbitCamera,
    /// In physical window pixels.
    pub viewport: Viewport,
}

impl ViewInfo {
    pub fn aspect(&self) -> f32 {
        self.viewport.width.max(1.0) / self.viewport.height.max(1.0)
    }

    pub fn view_proj(&self) -> Mat4 {
        self.camera.view_projection(self.aspect())
    }

    pub fn eye(&self) -> Vec3 {
        self.camera.eye()
    }

    /// Window pixel position of `p`, or `None` if it is behind the camera.
    pub fn project(&self, p: Vec3) -> Option<Vec2> {
        let clip = self.view_proj() * p.extend(1.0);
        if clip.w <= 1e-6 {
            return None;
        }
        let ndc = Vec2::new(clip.x, clip.y) / clip.w;
        let v = self.viewport;
        Some(Vec2::new(
            v.x + (ndc.x + 1.0) * 0.5 * v.width,
            v.y + (ndc.y + 1.0) * 0.5 * v.height,
        ))
    }

    /// Ray from the eye through a window pixel: origin and unit direction.
    pub fn ray(&self, cursor: Vec2) -> (Vec3, Vec3) {
        let v = self.viewport;
        let ndc = Vec2::new(
            (cursor.x - v.x) / v.width.max(1.0) * 2.0 - 1.0,
            (cursor.y - v.y) / v.height.max(1.0) * 2.0 - 1.0,
        );
        self.camera.ray(ndc, self.aspect())
    }

    /// Size of one pixel in world units at the depth of `p`.
    pub fn world_per_pixel_at(&self, p: Vec3) -> f32 {
        let depth = (p - self.eye()).dot(-self.camera.back()).max(1e-6);
        2.0 * depth * (self.camera.fov_y * 0.5).tan() / self.viewport.height.max(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_and_ray_agree() {
        let view = ViewInfo {
            camera: OrbitCamera {
                yaw: 0.5,
                pitch: 0.3,
                ..Default::default()
            },
            viewport: Viewport {
                x: 100.0,
                y: 20.0,
                width: 800.0,
                height: 600.0,
            },
        };
        let target_px = view.project(view.camera.target).unwrap();
        assert!((target_px - Vec2::new(500.0, 320.0)).length() < 1e-2);
        for px in [Vec2::new(150.0, 40.0), Vec2::new(850.0, 590.0)] {
            let (o, d) = view.ray(px);
            let back = view.project(o + d * 3.0).unwrap();
            assert!((back - px).length() < 1e-2, "{px} -> {back}");
        }
        let behind = view.eye() + view.camera.back();
        assert!(view.project(behind).is_none());
    }
}
