//! Mirror symmetry for retopology edits.
//!
//! Mirrored vertices are paired by position: the partner of a vertex is the
//! vertex at its exact reflection. Edits keep pairs exact (new partners are
//! placed at the exact reflection, not re-snapped), so the pairing survives
//! any sequence of edits, undo included, with no extra bookkeeping.

use polyflow_core::glam::Vec3;
use polyflow_core::halfedge::{HalfEdgeMesh, VertexId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Axis::X => "X",
            Axis::Y => "Y",
            Axis::Z => "Z",
        }
    }
}

/// A mirror plane through the origin, perpendicular to `axis`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Symmetry {
    pub axis: Axis,
    /// Vertices closer than this to the plane are on the seam: they are their
    /// own partner and stay on the plane.
    pub seam_tolerance: f32,
    /// Two positions closer than this are the same mirrored vertex.
    pub match_tolerance: f32,
}

impl Symmetry {
    /// Tolerances scaled to a scene of the given radius.
    pub fn new(axis: Axis, scene_radius: f32) -> Self {
        Self {
            axis,
            seam_tolerance: scene_radius * 5e-3,
            match_tolerance: scene_radius * 1e-5,
        }
    }

    pub fn reflect(&self, p: Vec3) -> Vec3 {
        let mut q = p;
        q[self.axis.index()] = -q[self.axis.index()];
        q
    }

    pub fn is_on_seam(&self, p: Vec3) -> bool {
        p[self.axis.index()].abs() <= self.seam_tolerance
    }

    /// `p` moved onto the plane if it is within the seam tolerance.
    pub fn snap_to_seam(&self, p: Vec3) -> Vec3 {
        let mut q = p;
        if self.is_on_seam(p) {
            q[self.axis.index()] = 0.0;
        }
        q
    }

    /// The mirror partner of `v`: itself on the seam, otherwise the vertex at
    /// its reflection, if any.
    pub fn partner(&self, mesh: &HalfEdgeMesh, v: VertexId) -> Option<VertexId> {
        let p = mesh.position(v);
        if self.is_on_seam(p) {
            return Some(v);
        }
        self.vertex_at(mesh, self.reflect(p))
    }

    /// The vertex at `p` (within the match tolerance), if any.
    pub fn vertex_at(&self, mesh: &HalfEdgeMesh, p: Vec3) -> Option<VertexId> {
        let tolerance2 = self.match_tolerance * self.match_tolerance;
        mesh.vertices()
            .map(|v| (v, mesh.position(v).distance_squared(p)))
            .filter(|&(_, d2)| d2 <= tolerance2)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(v, _)| v)
    }

    /// The partner of `v`, creating it at the exact reflection if missing.
    pub fn partner_or_create(&self, mesh: &mut HalfEdgeMesh, v: VertexId) -> VertexId {
        match self.partner(mesh, v) {
            Some(partner) => partner,
            None => {
                let p = self.reflect(mesh.position(v));
                mesh.add_vertex(p)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partners() {
        let sym = Symmetry::new(Axis::X, 1.0);
        let mut mesh = HalfEdgeMesh::new();
        let a = mesh.add_vertex(Vec3::new(0.5, 0.2, 0.1));
        let seam = mesh.add_vertex(Vec3::new(0.001, 0.3, 0.0));
        assert_eq!(sym.partner(&mesh, a), None);
        let b = sym.partner_or_create(&mut mesh, a);
        assert_eq!(mesh.position(b), Vec3::new(-0.5, 0.2, 0.1));
        assert_eq!(sym.partner(&mesh, a), Some(b));
        assert_eq!(sym.partner(&mesh, b), Some(a));
        assert_eq!(sym.partner(&mesh, seam), Some(seam));
        assert_eq!(sym.snap_to_seam(Vec3::new(0.001, 0.3, 0.0)).x, 0.0);
        assert_eq!(sym.partner_or_create(&mut mesh, a), b, "no duplicates");
    }
}
