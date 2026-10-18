//! Retopology: the editable mesh, its tools and how it is drawn.

pub mod edit;
pub mod geometry;
pub mod pick;
pub mod surface;
pub mod symmetry;
pub mod tools;
pub mod view;

use polyflow_core::halfedge::HalfEdgeMesh;
use polyflow_core::history::MeshDocument;
use polyflow_render::{Overlay, OverlayLine, OverlayPoint, OverlayVertex, Rgba8};

use self::symmetry::{Axis, Symmetry};

pub const FACE_COLOR: Rgba8 = Rgba8::new(70, 140, 255, 55);
pub const EDGE_COLOR: Rgba8 = Rgba8::new(150, 200, 255, 255);
pub const BOUNDARY_EDGE_COLOR: Rgba8 = Rgba8::new(255, 255, 255, 255);
pub const VERTEX_COLOR: Rgba8 = Rgba8::new(225, 238, 255, 255);
pub const HOVER_COLOR: Rgba8 = Rgba8::new(255, 214, 64, 255);
pub const HOVER_FACE_COLOR: Rgba8 = Rgba8::new(255, 214, 64, 70);
pub const PENDING_COLOR: Rgba8 = Rgba8::new(255, 140, 40, 255);
pub const VERTEX_SIZE: f32 = 7.0;

/// The retopology mesh with its undo history and editing settings.
#[derive(Debug)]
pub struct Retopo {
    pub doc: MeshDocument,
    /// Mirror plane, if symmetry is on.
    pub symmetry_axis: Option<Axis>,
    /// Draw (and pick) the retopo mesh through the reference surface.
    pub x_ray: bool,
    /// How far the retopo mesh is drawn in front of the surface, as a
    /// fraction of its distance to the camera.
    pub surface_offset: f32,
    /// Size of the reference mesh, which scales symmetry tolerances.
    pub scene_radius: f32,
    next_gesture: u64,
    mesh_overlay: Overlay,
    mesh_overlay_version: Option<u64>,
}

impl Default for Retopo {
    fn default() -> Self {
        Self {
            doc: MeshDocument::default(),
            symmetry_axis: None,
            x_ray: false,
            surface_offset: 0.01,
            scene_radius: 1.0,
            next_gesture: 0,
            mesh_overlay: Overlay::default(),
            mesh_overlay_version: None,
        }
    }
}

impl Retopo {
    pub fn mesh(&self) -> &HalfEdgeMesh {
        self.doc.mesh()
    }

    pub fn symmetry(&self) -> Option<Symmetry> {
        self.symmetry_axis
            .map(|axis| Symmetry::new(axis, self.scene_radius))
    }

    /// A fresh key for `MeshDocument::edit_merged`, one per gesture.
    pub fn new_gesture(&mut self) -> u64 {
        self.next_gesture += 1;
        self.next_gesture
    }

    /// The mesh as overlay geometry, rebuilt only when the mesh changed.
    pub fn mesh_overlay(&mut self) -> &Overlay {
        let version = self.doc.mesh().version();
        if self.mesh_overlay_version != Some(version) {
            build_mesh_overlay(self.doc.mesh(), &mut self.mesh_overlay);
            self.mesh_overlay_version = Some(version);
        }
        &self.mesh_overlay
    }

    /// Counts of faces by number of sides: triangles, quads, others.
    pub fn face_kinds(&self) -> [usize; 3] {
        let mesh = self.doc.mesh();
        let mut kinds = [0; 3];
        for f in mesh.faces() {
            match mesh.face_halfedges(f).count() {
                3 => kinds[0] += 1,
                4 => kinds[1] += 1,
                _ => kinds[2] += 1,
            }
        }
        kinds
    }
}

fn build_mesh_overlay(mesh: &HalfEdgeMesh, out: &mut Overlay) {
    out.clear();
    for f in mesh.faces() {
        let vertices: Vec<_> = mesh.face_vertices(f).map(|v| mesh.position(v)).collect();
        for i in 1..vertices.len() - 1 {
            for p in [vertices[0], vertices[i], vertices[i + 1]] {
                out.triangles.push(OverlayVertex {
                    position: p,
                    color: FACE_COLOR,
                });
            }
        }
    }
    for e in mesh.edges() {
        let h = e.halfedge(0);
        let color = if mesh.is_boundary_edge(e) {
            BOUNDARY_EDGE_COLOR
        } else {
            EDGE_COLOR
        };
        out.lines.push(OverlayLine {
            a: mesh.position(mesh.origin(h)),
            b: mesh.position(mesh.target(h)),
            color,
        });
    }
    for v in mesh.vertices() {
        out.points.push(OverlayPoint {
            position: mesh.position(v),
            color: VERTEX_COLOR,
            size: VERTEX_SIZE,
        });
    }
}
