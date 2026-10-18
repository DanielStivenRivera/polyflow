//! Geometry for tools: turning strokes into quad strips and relaxing vertices.

use std::collections::HashMap;

use polyflow_core::glam::Vec3;
use polyflow_core::halfedge::{HalfEdgeMesh, VertexId};

use super::surface::SurfacePoint;

/// Rows of a quad strip along a stroke: `[left, right]` positions per row,
/// before snapping. `spacing` is the distance between rows and `width` the
/// distance across; the stroke is resampled so rows are evenly spaced.
/// Returns no rows for strokes shorter than half a row.
pub fn strip_rows(stroke: &[SurfacePoint], width: f32, spacing: f32) -> Vec<[Vec3; 2]> {
    if stroke.len() < 2 || width <= 0.0 || spacing <= 0.0 {
        return Vec::new();
    }
    // Cumulative arc length along the stroke.
    let mut lengths = Vec::with_capacity(stroke.len());
    let mut total = 0.0;
    lengths.push(0.0);
    for w in stroke.windows(2) {
        total += w[0].position.distance(w[1].position);
        lengths.push(total);
    }
    if total < spacing * 0.5 {
        return Vec::new();
    }
    let segments = (total / spacing).round().max(1.0) as usize;

    // Evenly spaced samples, interpolating position and normal.
    let mut samples = Vec::with_capacity(segments + 1);
    let mut j = 0;
    for k in 0..=segments {
        let s = total * k as f32 / segments as f32;
        while j + 2 < lengths.len() && lengths[j + 1] < s {
            j += 1;
        }
        let span = (lengths[j + 1] - lengths[j]).max(f32::MIN_POSITIVE);
        let t = ((s - lengths[j]) / span).clamp(0.0, 1.0);
        let (a, b) = (stroke[j], stroke[j + 1]);
        samples.push(SurfacePoint {
            position: a.position.lerp(b.position, t),
            normal: a.normal.lerp(b.normal, t).normalize_or(a.normal),
        });
    }

    let half = width * 0.5;
    (0..samples.len())
        .map(|k| {
            let prev = samples[k.saturating_sub(1)].position;
            let next = samples[(k + 1).min(samples.len() - 1)].position;
            let tangent = (next - prev).normalize_or(Vec3::X);
            let s = samples[k];
            // "Left" as seen from the surface normal.
            let side = s.normal.cross(tangent).normalize_or(Vec3::Y);
            [s.position + side * half, s.position - side * half]
        })
        .collect()
}

/// Laplacian relaxation of `vertices`: each moves towards the average of its
/// neighbors by `weights[i]` (0..1). Boundary vertices stay put unless
/// `relax_borders`; then they only average with boundary neighbors, so
/// borders smooth along themselves instead of being pulled into the
/// interior. Returns the new positions; the mesh is not modified.
pub fn relaxed_positions(
    mesh: &HalfEdgeMesh,
    vertices: &[VertexId],
    weights: &[f32],
    relax_borders: bool,
) -> HashMap<VertexId, Vec3> {
    vertices
        .iter()
        .zip(weights)
        .map(|(&v, &w)| {
            let p = mesh.position(v);
            let boundary = mesh.is_boundary_vertex(v);
            if boundary && !relax_borders {
                return (v, p);
            }
            let neighbors: Vec<Vec3> = mesh
                .outgoing_halfedges(v)
                .filter(|&h| !boundary || mesh.is_boundary_edge(h.edge()))
                .map(|h| mesh.position(mesh.target(h)))
                .collect();
            if neighbors.is_empty() {
                return (v, p);
            }
            let average = neighbors.iter().sum::<Vec3>() / neighbors.len() as f32;
            (v, p.lerp(average, w.clamp(0.0, 1.0)))
        })
        .collect()
}

/// Smooth falloff from 1 at the center to 0 at `x = 1`.
pub fn falloff(x: f32) -> f32 {
    let t = (1.0 - x * x).clamp(0.0, 1.0);
    t * t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_stroke(points: &[(f32, f32)]) -> Vec<SurfacePoint> {
        points
            .iter()
            .map(|&(x, z)| SurfacePoint {
                position: Vec3::new(x, 0.0, z),
                normal: Vec3::Y,
            })
            .collect()
    }

    #[test]
    fn straight_strip() {
        // Unevenly sampled stroke along +X on the XZ plane.
        let stroke = flat_stroke(&[(0.0, 0.0), (0.3, 0.0), (0.35, 0.0), (1.0, 0.0), (2.0, 0.0)]);
        let rows = strip_rows(&stroke, 0.5, 0.5);
        assert_eq!(rows.len(), 5);
        for (k, [left, right]) in rows.iter().enumerate() {
            let x = k as f32 * 0.5;
            // normal (+Y) x tangent (+X) = -Z.
            assert!(left.distance(Vec3::new(x, 0.0, -0.25)) < 1e-5, "{left}");
            assert!(right.distance(Vec3::new(x, 0.0, 0.25)) < 1e-5, "{right}");
        }
    }

    #[test]
    fn short_strokes_make_nothing() {
        assert!(strip_rows(&flat_stroke(&[(0.0, 0.0), (0.1, 0.0)]), 0.5, 0.5).is_empty());
        assert!(strip_rows(&flat_stroke(&[(0.0, 0.0)]), 0.5, 0.5).is_empty());
    }

    #[test]
    fn relaxing_evens_out_a_grid() {
        let mut mesh = HalfEdgeMesh::new();
        let mut ids = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                ids.push(mesh.add_vertex(Vec3::new(i as f32, j as f32, 0.0)));
            }
        }
        for (a, b, c, d) in [(0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7)] {
            mesh.add_face(&[ids[a], ids[b], ids[c], ids[d]]).unwrap();
        }
        let center = ids[4];
        mesh.set_position(center, Vec3::new(1.6, 0.7, 0.0));
        let corner = ids[0];
        let moved = relaxed_positions(&mesh, &[center, corner], &[1.0, 1.0], false);
        assert!(moved[&center].distance(Vec3::new(1.0, 1.0, 0.0)) < 1e-6);
        assert_eq!(moved[&corner], Vec3::ZERO, "borders are pinned by default");
        let moved = relaxed_positions(&mesh, &[corner], &[1.0], true);
        // The corner only averages its two boundary neighbors.
        assert!(moved[&corner].distance(Vec3::new(0.5, 0.5, 0.0)) < 1e-6);
        assert_eq!(falloff(0.0), 1.0);
        assert_eq!(falloff(1.0), 0.0);
    }
}
