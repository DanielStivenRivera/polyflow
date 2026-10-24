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
/// Faces with UVs under the UV checker (only the alpha is used).
pub const UV_FACE_COLOR: Rgba8 = Rgba8::new(255, 255, 255, 200);
/// Faces without UVs while the UV checker is on.
pub const NO_UV_FACE_COLOR: Rgba8 = Rgba8::new(255, 60, 200, 110);
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
    /// Shade faces with a checker from their UVs.
    pub uv_checker: bool,
    /// Size of the reference mesh, which scales symmetry tolerances.
    pub scene_radius: f32,
    next_gesture: u64,
    mesh_overlay: Overlay,
    /// Mesh version and UV checker state the overlay was built for.
    mesh_overlay_key: Option<(u64, bool)>,
    /// Stats and the mesh version they were counted for.
    stats: Option<(u64, RetopoStats)>,
}

/// Face counts shown in the retopology panel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetopoStats {
    /// Triangles, quads and other polygons.
    pub face_kinds: [usize; 3],
    /// Faces with a UV on every corner.
    pub uv_faces: usize,
}

impl Default for Retopo {
    fn default() -> Self {
        Self {
            doc: MeshDocument::default(),
            symmetry_axis: None,
            x_ray: false,
            surface_offset: 0.01,
            uv_checker: false,
            scene_radius: 1.0,
            next_gesture: 0,
            mesh_overlay: Overlay::default(),
            mesh_overlay_key: None,
            stats: None,
        }
    }
}

impl Retopo {
    pub fn mesh(&self) -> &HalfEdgeMesh {
        self.doc.mesh()
    }

    /// Replaces the mesh and its history (e.g. after importing a file).
    pub fn replace_mesh(&mut self, mesh: HalfEdgeMesh) {
        self.doc = MeshDocument::new(mesh);
        self.mesh_overlay_key = None;
        self.stats = None;
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

    /// The mesh as overlay geometry, rebuilt only when the mesh or the way
    /// it is shown changed.
    pub fn mesh_overlay(&mut self) -> &Overlay {
        let key = (self.doc.mesh().version(), self.uv_checker);
        if self.mesh_overlay_key != Some(key) {
            build_mesh_overlay(self.doc.mesh(), self.uv_checker, &mut self.mesh_overlay);
            self.mesh_overlay_key = Some(key);
        }
        &self.mesh_overlay
    }

    /// Face counts, recounted only when the mesh changed (the panel asks
    /// every frame, and a retopology can have millions of faces).
    pub fn stats(&mut self) -> RetopoStats {
        let mesh = self.doc.mesh();
        let version = mesh.version();
        if let Some((counted, stats)) = self.stats
            && counted == version
        {
            return stats;
        }
        let mut stats = RetopoStats::default();
        for f in mesh.faces() {
            let (sides, uvs) = mesh.face_halfedges(f).fold((0, true), |(n, uvs), h| {
                (n + 1, uvs && mesh.uv(h).is_some())
            });
            match sides {
                3 => stats.face_kinds[0] += 1,
                4 => stats.face_kinds[1] += 1,
                _ => stats.face_kinds[2] += 1,
            }
            stats.uv_faces += uvs as usize;
        }
        self.stats = Some((version, stats));
        stats
    }
}

fn build_mesh_overlay(mesh: &HalfEdgeMesh, uv_checker: bool, out: &mut Overlay) {
    out.clear();
    let mut corners: Vec<OverlayVertex> = Vec::new();
    for f in mesh.faces() {
        let checker = uv_checker && mesh.face_has_uvs(f);
        corners.clear();
        corners.extend(mesh.face_halfedges(f).map(|h| {
            let p = mesh.position(mesh.origin(h));
            match mesh.uv(h) {
                Some(uv) if checker => OverlayVertex::checker(p, UV_FACE_COLOR, uv),
                _ if uv_checker => OverlayVertex::flat(p, NO_UV_FACE_COLOR),
                _ => OverlayVertex::flat(p, FACE_COLOR),
            }
        }));
        for i in 1..corners.len() - 1 {
            out.triangles
                .extend_from_slice(&[corners[0], corners[i], corners[i + 1]]);
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
