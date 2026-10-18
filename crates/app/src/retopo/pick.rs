//! Finding retopology elements under the cursor.

use polyflow_core::glam::{Vec2, Vec3};
use polyflow_core::halfedge::{FaceId, HalfEdgeMesh, VertexId};

use super::surface::Surface;
use super::view::ViewInfo;

/// How many of the closest candidates are tested for visibility (each test
/// is a raycast) before giving up.
const VISIBILITY_CANDIDATES: usize = 8;

/// The visible vertex closest to `cursor`, within `radius` pixels. With
/// `x_ray`, hidden vertices count too.
pub fn vertex(
    mesh: &HalfEdgeMesh,
    surface: &Surface<'_>,
    view: &ViewInfo,
    cursor: Vec2,
    radius: f32,
    x_ray: bool,
) -> Option<VertexId> {
    let mut candidates: Vec<(f32, VertexId)> = mesh
        .vertices()
        .filter_map(|v| {
            let d = view.project(mesh.position(v))?.distance(cursor);
            (d <= radius).then_some((d, v))
        })
        .collect();
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    let eye = view.eye();
    candidates
        .into_iter()
        .take(VISIBILITY_CANDIDATES)
        .find(|&(_, v)| x_ray || surface.is_visible(eye, mesh.position(v)))
        .map(|(_, v)| v)
}

/// The visible face under `cursor`, nearest to the camera.
pub fn face(
    mesh: &HalfEdgeMesh,
    surface: &Surface<'_>,
    view: &ViewInfo,
    cursor: Vec2,
    x_ray: bool,
) -> Option<FaceId> {
    let eye = view.eye();
    let mut hits: Vec<(f32, FaceId, Vec3)> = mesh
        .faces()
        .filter_map(|f| {
            let positions: Vec<Vec3> = mesh.face_vertices(f).map(|v| mesh.position(v)).collect();
            let screen: Option<Vec<Vec2>> = positions.iter().map(|&p| view.project(p)).collect();
            if !point_in_polygon(cursor, &screen?) {
                return None;
            }
            let centroid = positions.iter().sum::<Vec3>() / positions.len() as f32;
            Some((centroid.distance_squared(eye), f, centroid))
        })
        .collect();
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    hits.into_iter()
        .take(VISIBILITY_CANDIDATES)
        .find(|&(_, _, c)| x_ray || surface.is_visible(eye, c))
        .map(|(_, f, _)| f)
}

/// Even-odd test; works for either winding.
fn point_in_polygon(p: Vec2, polygon: &[Vec2]) -> bool {
    let mut inside = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let (a, b) = (polygon[i], polygon[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polygon_containment() {
        let square = [
            Vec2::ZERO,
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        assert!(point_in_polygon(Vec2::new(5.0, 5.0), &square));
        assert!(!point_in_polygon(Vec2::new(15.0, 5.0), &square));
        let reversed: Vec<Vec2> = square.iter().rev().copied().collect();
        assert!(point_in_polygon(Vec2::new(1.0, 9.0), &reversed));
    }
}
