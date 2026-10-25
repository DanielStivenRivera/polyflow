//! Retopology: the editable mesh, its tools and how it is drawn.

pub mod edit;
pub mod geometry;
pub mod pick;
pub mod selection;
pub mod surface;
pub mod symmetry;
pub mod tools;
pub mod view;

use polyflow_bake::Cage;
use polyflow_core::halfedge::HalfEdgeMesh;
use polyflow_core::history::MeshDocument;
use polyflow_render::{Overlay, OverlayLine, OverlayPoint, OverlayVertex, Rgba8};

use self::selection::Selection;
use self::symmetry::{Axis, Symmetry};

pub const FACE_COLOR: Rgba8 = Rgba8::new(70, 140, 255, 55);
/// Faces with UVs under the UV checker (only the alpha is used).
pub const UV_FACE_COLOR: Rgba8 = Rgba8::new(255, 255, 255, 200);
/// Faces without UVs while the UV checker is on.
pub const NO_UV_FACE_COLOR: Rgba8 = Rgba8::new(255, 60, 200, 110);
pub const EDGE_COLOR: Rgba8 = Rgba8::new(150, 200, 255, 255);
pub const BOUNDARY_EDGE_COLOR: Rgba8 = Rgba8::new(255, 255, 255, 255);
/// Edges marked hard for baking.
pub const HARD_EDGE_COLOR: Rgba8 = Rgba8::new(80, 235, 255, 255);
pub const VERTEX_COLOR: Rgba8 = Rgba8::new(225, 238, 255, 255);
pub const HOVER_COLOR: Rgba8 = Rgba8::new(255, 214, 64, 255);
pub const HOVER_FACE_COLOR: Rgba8 = Rgba8::new(255, 214, 64, 70);
pub const PENDING_COLOR: Rgba8 = Rgba8::new(255, 140, 40, 255);
pub const SELECTED_COLOR: Rgba8 = Rgba8::new(255, 150, 30, 255);
pub const SELECTED_FACE_COLOR: Rgba8 = Rgba8::new(255, 150, 30, 90);
pub const VERTEX_SIZE: f32 = 7.0;
pub const CAGE_COLOR: Rgba8 = Rgba8::new(140, 235, 90, 210);
/// Default push of the automatic cage, as a fraction of the sculpt's radius
/// (the bake's default ray distance).
pub const DEFAULT_CAGE_PUSH: f32 = 0.03;

/// The retopology mesh with its undo history and editing settings.
#[derive(Debug)]
pub struct Retopo {
    pub doc: MeshDocument,
    pub selection: Selection,
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
    /// Draw the bake cage.
    pub show_cage: bool,
    /// Push of the automatic cage in scene units; `None` for the default.
    pub cage_push: Option<f32>,
    /// A cage imported from a file, used instead of the automatic one while
    /// it still fits the mesh.
    imported_cage: Option<Cage>,
    /// Bumped whenever `imported_cage` changes.
    imported_cage_generation: u64,
    cage_overlay: Overlay,
    /// Mesh version, push and imported cage the cage overlay was built for.
    cage_overlay_key: Option<(u64, u32, u64)>,
    next_gesture: u64,
    mesh_overlay: Overlay,
    /// Mesh version and UV checker state the overlay was built for.
    mesh_overlay_key: Option<(u64, bool)>,
    selection_overlay: Overlay,
    /// Mesh version and selection generation the overlay was built for.
    selection_overlay_key: Option<(u64, u64)>,
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
            selection: Selection::default(),
            symmetry_axis: None,
            x_ray: false,
            surface_offset: 0.01,
            uv_checker: false,
            scene_radius: 1.0,
            show_cage: false,
            cage_push: None,
            imported_cage: None,
            imported_cage_generation: 0,
            cage_overlay: Overlay::default(),
            cage_overlay_key: None,
            next_gesture: 0,
            mesh_overlay: Overlay::default(),
            mesh_overlay_key: None,
            selection_overlay: Overlay::default(),
            selection_overlay_key: None,
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
        self.selection.clear();
        self.mesh_overlay_key = None;
        self.stats = None;
        self.set_imported_cage(None);
    }

    /// Runs an edit as one undo step (see `MeshDocument::edit`), keeping
    /// the selection to the elements it did not remove or rebuild.
    pub fn edit<R, E>(
        &mut self,
        name: &str,
        edit: impl FnOnce(&mut HalfEdgeMesh) -> Result<R, E>,
    ) -> Result<R, E> {
        let snapshot = self.selection.snapshot(self.doc.mesh());
        let result = self.doc.edit(name, edit);
        self.selection.revalidate(self.doc.mesh(), &snapshot);
        result
    }

    /// [`Self::edit`], merged into the gesture's undo step (see
    /// `MeshDocument::edit_merged`).
    pub fn edit_merged<R, E>(
        &mut self,
        name: &str,
        gesture: u64,
        edit: impl FnOnce(&mut HalfEdgeMesh) -> Result<R, E>,
    ) -> Result<R, E> {
        let snapshot = self.selection.snapshot(self.doc.mesh());
        let result = self.doc.edit_merged(name, gesture, edit);
        self.selection.revalidate(self.doc.mesh(), &snapshot);
        result
    }

    pub fn undo(&mut self) {
        let snapshot = self.selection.snapshot(self.doc.mesh());
        self.doc.undo();
        self.selection.revalidate(self.doc.mesh(), &snapshot);
    }

    pub fn redo(&mut self) {
        let snapshot = self.selection.snapshot(self.doc.mesh());
        self.doc.redo();
        self.selection.revalidate(self.doc.mesh(), &snapshot);
    }

    /// The automatic cage's push in scene units.
    pub fn cage_push(&self) -> f32 {
        self.cage_push
            .unwrap_or(self.scene_radius * DEFAULT_CAGE_PUSH)
    }

    pub fn imported_cage(&self) -> Option<&Cage> {
        self.imported_cage.as_ref()
    }

    pub fn set_imported_cage(&mut self, cage: Option<Cage>) {
        self.imported_cage = cage;
        self.imported_cage_generation += 1;
    }

    /// Whether the imported cage no longer fits (vertices were added or
    /// removed since), so the automatic cage is used instead.
    pub fn imported_cage_is_stale(&self) -> bool {
        self.imported_cage
            .as_ref()
            .is_some_and(|c| !c.fits(self.doc.mesh()))
    }

    /// The cage a bake would use: the imported one if it fits, else the
    /// automatic one.
    pub fn cage(&self) -> Cage {
        match &self.imported_cage {
            Some(cage) if cage.fits(self.doc.mesh()) => cage.clone(),
            _ => Cage::automatic(self.doc.mesh(), self.cage_push()),
        }
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

    /// Drops selected elements that no longer exist, and rebuilds the
    /// overlays that are out of date. Call once per frame before
    /// [`Self::mesh_overlay`], [`Self::selection_overlay`] and
    /// [`Self::cage_overlay`].
    pub fn update_overlays(&mut self) {
        let version = self.doc.mesh().version();
        if self.selection_overlay_key.is_none_or(|(v, _)| v != version) {
            self.selection.prune(self.doc.mesh());
        }
        let key = (version, self.selection.generation());
        if self.selection_overlay_key != Some(key) {
            build_selection_overlay(
                self.doc.mesh(),
                &self.selection,
                &mut self.selection_overlay,
            );
            self.selection_overlay_key = Some(key);
        }
        let key = (version, self.uv_checker);
        if self.mesh_overlay_key != Some(key) {
            build_mesh_overlay(self.doc.mesh(), self.uv_checker, &mut self.mesh_overlay);
            self.mesh_overlay_key = Some(key);
        }
        if self.show_cage {
            let key = (
                version,
                self.cage_push().to_bits(),
                self.imported_cage_generation,
            );
            if self.cage_overlay_key != Some(key) {
                build_cage_overlay(self.doc.mesh(), &self.cage(), &mut self.cage_overlay);
                self.cage_overlay_key = Some(key);
            }
        }
    }

    /// The mesh as overlay geometry (see [`Self::update_overlays`]).
    pub fn mesh_overlay(&self) -> &Overlay {
        &self.mesh_overlay
    }

    /// The selected elements, highlighted (see [`Self::update_overlays`]).
    pub fn selection_overlay(&self) -> &Overlay {
        &self.selection_overlay
    }

    /// The cage's wireframe, if it is shown (see [`Self::update_overlays`]).
    pub fn cage_overlay(&self) -> Option<&Overlay> {
        self.show_cage.then_some(&self.cage_overlay)
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

fn build_cage_overlay(mesh: &HalfEdgeMesh, cage: &Cage, out: &mut Overlay) {
    out.clear();
    for e in mesh.edges() {
        let h = e.halfedge(0);
        let at = |v| cage.position(v).unwrap_or_else(|| mesh.position(v));
        out.lines.push(OverlayLine {
            a: at(mesh.origin(h)),
            b: at(mesh.target(h)),
            color: CAGE_COLOR,
        });
    }
}

fn build_selection_overlay(mesh: &HalfEdgeMesh, selection: &Selection, out: &mut Overlay) {
    out.clear();
    for &f in selection.faces() {
        let p: Vec<_> = mesh.face_vertices(f).map(|v| mesh.position(v)).collect();
        for i in 1..p.len() - 1 {
            for q in [p[0], p[i], p[i + 1]] {
                out.triangles
                    .push(OverlayVertex::flat(q, SELECTED_FACE_COLOR));
            }
        }
    }
    let edges = selection.edges().iter().copied().chain(
        selection
            .faces()
            .iter()
            .flat_map(|&f| mesh.face_halfedges(f).map(|h| h.edge()).collect::<Vec<_>>()),
    );
    for e in edges {
        let h = e.halfedge(0);
        out.lines.push(OverlayLine {
            a: mesh.position(mesh.origin(h)),
            b: mesh.position(mesh.target(h)),
            color: SELECTED_COLOR,
        });
    }
    for &v in selection.vertices() {
        out.points.push(OverlayPoint {
            position: mesh.position(v),
            color: SELECTED_COLOR,
            size: VERTEX_SIZE + 2.0,
        });
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
        let color = if mesh.is_hard(e) {
            HARD_EDGE_COLOR
        } else if mesh.is_boundary_edge(e) {
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
