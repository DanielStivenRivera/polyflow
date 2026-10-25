//! Finding retopology elements under the cursor.

use polyflow_core::glam::{Vec2, Vec3};
use polyflow_core::halfedge::{EdgeId, FaceId, HalfEdgeId, HalfEdgeMesh, VertexId};

use super::selection::{Elements, SelectMode};
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
    vertex_except(mesh, surface, view, cursor, radius, x_ray, None)
}

/// [`vertex`], ignoring `except`.
pub fn vertex_except(
    mesh: &HalfEdgeMesh,
    surface: &Surface<'_>,
    view: &ViewInfo,
    cursor: Vec2,
    radius: f32,
    x_ray: bool,
    except: Option<VertexId>,
) -> Option<VertexId> {
    let mut candidates: Vec<(f32, VertexId)> = mesh
        .vertices()
        .filter(|&v| Some(v) != except)
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

/// The visible edge closest to `cursor`, within `radius` pixels.
pub fn edge(
    mesh: &HalfEdgeMesh,
    surface: &Surface<'_>,
    view: &ViewInfo,
    cursor: Vec2,
    radius: f32,
    x_ray: bool,
) -> Option<EdgeId> {
    let mut candidates: Vec<(f32, EdgeId, Vec3)> = mesh
        .edges()
        .filter_map(|e| {
            let h = e.halfedge(0);
            let (a, b) = (mesh.position(mesh.origin(h)), mesh.position(mesh.target(h)));
            let (pa, pb) = (view.project(a)?, view.project(b)?);
            let t = closest_on_segment(cursor, pa, pb);
            let d = pa.lerp(pb, t).distance(cursor);
            (d <= radius).then(|| (d, e, a.lerp(b, t)))
        })
        .collect();
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    let eye = view.eye();
    candidates
        .into_iter()
        .take(VISIBILITY_CANDIDATES)
        .find(|&(_, _, p)| x_ray || surface.is_visible(eye, p))
        .map(|(_, e, _)| e)
}

/// Where along `h` (0 at its origin, 1 at its target) the cursor is, as seen
/// on screen.
pub fn halfedge_param(mesh: &HalfEdgeMesh, view: &ViewInfo, h: HalfEdgeId, cursor: Vec2) -> f32 {
    let a = view.project(mesh.position(mesh.origin(h)));
    let b = view.project(mesh.position(mesh.target(h)));
    match (a, b) {
        (Some(a), Some(b)) => closest_on_segment(cursor, a, b),
        _ => 0.5,
    }
}

/// A screen area to select in, in window pixels.
#[derive(Debug, Clone, PartialEq)]
pub enum Region {
    /// Two opposite corners.
    Box(Vec2, Vec2),
    /// A closed outline.
    Lasso(Vec<Vec2>),
}

impl Region {
    pub fn contains(&self, p: Vec2) -> bool {
        match self {
            Region::Box(a, b) => p.cmpge(a.min(*b)).all() && p.cmple(a.max(*b)).all(),
            Region::Lasso(outline) => outline.len() >= 3 && point_in_polygon(p, outline),
        }
    }
}

/// The visible elements of `mode` inside `region`: vertices inside it, edges
/// with both ends inside, faces with their center inside.
pub fn in_region(
    mesh: &HalfEdgeMesh,
    surface: &Surface<'_>,
    view: &ViewInfo,
    region: &Region,
    mode: SelectMode,
    x_ray: bool,
) -> Elements {
    let eye = view.eye();
    let inside = |p: Vec3| view.project(p).is_some_and(|px| region.contains(px));
    let visible = |p: Vec3| x_ray || surface.is_visible(eye, p);
    match mode {
        SelectMode::Vertex => Elements::Vertices(
            mesh.vertices()
                .filter(|&v| {
                    let p = mesh.position(v);
                    inside(p) && visible(p)
                })
                .collect(),
        ),
        SelectMode::Edge => Elements::Edges(
            mesh.edges()
                .filter(|&e| {
                    let h = e.halfedge(0);
                    let (a, b) = (mesh.position(mesh.origin(h)), mesh.position(mesh.target(h)));
                    inside(a) && inside(b) && visible((a + b) * 0.5)
                })
                .collect(),
        ),
        SelectMode::Face => Elements::Faces(
            mesh.faces()
                .filter(|&f| {
                    let (sum, n) = mesh
                        .face_vertices(f)
                        .fold((Vec3::ZERO, 0), |(s, n), v| (s + mesh.position(v), n + 1));
                    let center = sum / n as f32;
                    inside(center) && visible(center)
                })
                .collect(),
        ),
    }
}

/// Distance in pixels from `p` to the segment `ab`.
pub fn distance_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    a.lerp(b, closest_on_segment(p, a, b)).distance(p)
}

/// Parameter in 0..=1 of the point of segment `ab` closest to `p`.
fn closest_on_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 <= 1e-12 {
        return 0.0;
    }
    ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
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

    #[test]
    fn regions() {
        let r = Region::Box(Vec2::new(10.0, 10.0), Vec2::new(0.0, 0.0));
        assert!(r.contains(Vec2::new(5.0, 5.0)));
        assert!(!r.contains(Vec2::new(5.0, 11.0)));
        let lasso = Region::Lasso(vec![Vec2::ZERO, Vec2::new(10.0, 0.0), Vec2::new(0.0, 10.0)]);
        assert!(lasso.contains(Vec2::new(2.0, 2.0)));
        assert!(!lasso.contains(Vec2::new(8.0, 8.0)));
        assert!(!Region::Lasso(vec![Vec2::ZERO, Vec2::ONE]).contains(Vec2::ZERO));
        assert_eq!(
            closest_on_segment(Vec2::new(5.0, 3.0), Vec2::ZERO, Vec2::new(10.0, 0.0)),
            0.5
        );
    }
}
