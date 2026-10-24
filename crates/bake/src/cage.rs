//! Cages: an envelope around the low-poly that bake rays start from.
//!
//! A cage has the low-poly's vertices, moved. A texel's ray starts at the
//! texel's point on the cage and travels towards (and through) its point on
//! the low-poly, so the cage decides where every ray comes from: inflated
//! around ears and lips to catch them whole, pulled in where nearby geometry
//! would be caught by mistake.
//!
//! - [`Cage::automatic`] pushes each vertex along its smooth normal, the same
//!   normals the baker uses, so the cage stays closed across UV seams.
//! - [`Cage::from_mesh`] takes a cage edited elsewhere: the low-poly exported
//!   by Polyflow, with vertices moved but none added, removed or reordered.

use polyflow_core::glam::Vec3;
use polyflow_core::halfedge::{HalfEdgeMesh, VertexId};

use crate::mesh::vertex_normals;

/// Cage positions for a retopology's vertices.
#[derive(Debug, Clone, PartialEq)]
pub struct Cage {
    /// By vertex slot; `None` for free slots.
    positions: Vec<Option<Vec3>>,
}

/// Why an imported cage does not fit the retopology.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CageMismatch {
    #[error("the cage has {cage} vertices, the retopology {retopo}")]
    VertexCount { cage: usize, retopo: usize },
    #[error("the cage has {cage} faces, the retopology {retopo}")]
    FaceCount { cage: usize, retopo: usize },
    #[error(
        "the cage's faces connect different vertices than the retopology's \
         (were vertices reordered, added or removed?)"
    )]
    Faces,
}

impl Cage {
    /// Every vertex pushed `distance` along its smooth normal.
    pub fn automatic(mesh: &HalfEdgeMesh, distance: f32) -> Self {
        let normals = vertex_normals(mesh);
        let positions = mesh
            .vertex_slots()
            .zip(normals)
            .map(|(p, n)| p.map(|p| p + n * distance))
            .collect();
        Self { positions }
    }

    /// The cage `cage` (a copy of `mesh` with moved vertices, as read back
    /// from an OBJ written by Polyflow) for `mesh`. Vertices are matched in
    /// file order, and the faces must connect the same vertices.
    pub fn from_mesh(mesh: &HalfEdgeMesh, cage: &HalfEdgeMesh) -> Result<Self, CageMismatch> {
        if cage.vertex_count() != mesh.vertex_count() {
            return Err(CageMismatch::VertexCount {
                cage: cage.vertex_count(),
                retopo: mesh.vertex_count(),
            });
        }
        if cage.face_count() != mesh.face_count() {
            return Err(CageMismatch::FaceCount {
                cage: cage.face_count(),
                retopo: mesh.face_count(),
            });
        }
        if face_signature(mesh) != face_signature(cage) {
            return Err(CageMismatch::Faces);
        }
        let mut positions = vec![None; mesh.vertex_slots().count()];
        for (v, c) in mesh.vertices().zip(cage.vertices()) {
            positions[v.index()] = Some(cage.position(c));
        }
        Ok(Self { positions })
    }

    pub fn position(&self, v: VertexId) -> Option<Vec3> {
        self.positions.get(v.index()).copied().flatten()
    }

    /// Whether the cage has a position for exactly `mesh`'s vertices; edits
    /// that add or remove vertices make an imported cage stale.
    pub fn fits(&self, mesh: &HalfEdgeMesh) -> bool {
        let slots = mesh.vertex_slots().count();
        mesh.vertex_slots()
            .zip(self.positions.iter().chain(std::iter::repeat(&None)))
            .all(|(v, c)| v.is_some() == c.is_some())
            && self.positions[slots.min(self.positions.len())..]
                .iter()
                .all(Option::is_none)
    }

    /// A copy of `mesh` with the cage's positions, e.g. to save it as OBJ.
    /// Vertices the cage does not cover keep theirs.
    pub fn apply(&self, mesh: &HalfEdgeMesh) -> HalfEdgeMesh {
        let mut out = mesh.clone();
        let vertices: Vec<VertexId> = out.vertices().collect();
        for v in vertices {
            if let Some(p) = self.position(v) {
                out.set_position(v, p);
            }
        }
        out
    }
}

/// Faces as lists of dense vertex numbers (file order), rotated to start at
/// their lowest, and sorted: equal for meshes with the same connectivity.
fn face_signature(mesh: &HalfEdgeMesh) -> Vec<Vec<u32>> {
    let mut number = vec![u32::MAX; mesh.vertex_slots().count()];
    for (i, v) in mesh.vertices().enumerate() {
        number[v.index()] = i as u32;
    }
    let mut faces: Vec<Vec<u32>> = mesh
        .faces()
        .map(|f| {
            let mut face: Vec<u32> = mesh.face_vertices(f).map(|v| number[v.index()]).collect();
            let start = face
                .iter()
                .enumerate()
                .min_by_key(|&(_, n)| n)
                .map_or(0, |(i, _)| i);
            face.rotate_left(start);
            face
        })
        .collect();
    faces.sort_unstable();
    faces
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::tests::uv_grid;

    #[test]
    fn automatic_cages_follow_the_normals() {
        let mesh = uv_grid(2, false);
        let cage = Cage::automatic(&mesh, 0.25);
        assert!(cage.fits(&mesh));
        for v in mesh.vertices() {
            let expected = mesh.position(v) + Vec3::Z * 0.25;
            assert!(cage.position(v).unwrap().distance(expected) < 1e-6);
        }
    }

    #[test]
    fn imported_cages_must_match_the_connectivity() {
        let mesh = uv_grid(2, false);
        let moved = Cage::automatic(&mesh, 0.5).apply(&mesh);
        let cage = Cage::from_mesh(&mesh, &moved).unwrap();
        assert_eq!(cage, Cage::automatic(&mesh, 0.5));

        let bigger = uv_grid(3, false);
        assert!(matches!(
            Cage::from_mesh(&mesh, &bigger),
            Err(CageMismatch::VertexCount { .. })
        ));
        let mut fewer = mesh.clone();
        let f = fewer.faces().next().unwrap();
        fewer.remove_face(f, false).unwrap();
        assert!(matches!(
            Cage::from_mesh(&mesh, &fewer),
            Err(CageMismatch::FaceCount { .. })
        ));
        // Same vertices and counts, but faces swap vertices 0 and 1.
        let mut rewired = HalfEdgeMesh::new();
        let new: Vec<VertexId> = mesh
            .vertices()
            .map(|v| rewired.add_vertex(mesh.position(v)))
            .collect();
        let swap = |i: usize| match i {
            0 => 1,
            1 => 0,
            i => i,
        };
        for f in mesh.faces() {
            let face: Vec<VertexId> = mesh
                .face_vertices(f)
                .map(|v| new[swap(v.index())])
                .collect();
            rewired.add_face(&face).unwrap();
        }
        assert_eq!(Cage::from_mesh(&mesh, &rewired), Err(CageMismatch::Faces));
    }

    #[test]
    fn cages_go_stale_when_vertices_change() {
        let mut mesh = uv_grid(2, false);
        let cage = Cage::automatic(&mesh, 0.1);
        let v = mesh.vertices().next().unwrap();
        mesh.set_position(v, Vec3::ONE);
        assert!(cage.fits(&mesh), "moving vertices keeps the cage");
        mesh.add_vertex(Vec3::ZERO);
        assert!(!cage.fits(&mesh));
        let mut mesh = uv_grid(2, false);
        mesh.remove_vertex(v).unwrap();
        assert!(!cage.fits(&mesh));
    }
}
