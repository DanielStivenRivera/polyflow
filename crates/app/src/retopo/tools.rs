//! Interactive retopology tools.
//!
//! Tools receive mouse and keyboard events that the camera did not take
//! (left and right buttons without Alt) and edit the retopo mesh through
//! `MeshDocument`, so every action is one atomic undo step.

use std::collections::HashMap;

use polyflow_core::glam::{Vec2, Vec3};
use polyflow_core::halfedge::{FaceId, VertexId};
use polyflow_render::{Overlay, OverlayLine, OverlayPoint, OverlayVertex};
use winit::event::MouseButton;
use winit::keyboard::{Key, ModifiersState, NamedKey};

use super::surface::{Surface, SurfacePoint};
use super::view::ViewInfo;
use super::{
    HOVER_COLOR, HOVER_FACE_COLOR, PENDING_COLOR, Retopo, VERTEX_SIZE, edit, geometry, pick,
};

/// Pixels within which a vertex counts as under the cursor.
pub const PICK_RADIUS: f32 = 10.0;
/// Pixels the cursor may move between press and release for a click.
const CLICK_SLOP: f32 = 4.0;
/// Pixels between stroke samples.
const STROKE_STEP: f32 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Create,
    Draw,
    Relax,
}

impl ToolKind {
    pub const ALL: [ToolKind; 3] = [ToolKind::Create, ToolKind::Draw, ToolKind::Relax];

    pub fn label(self) -> &'static str {
        match self {
            ToolKind::Create => "Create",
            ToolKind::Draw => "Draw",
            ToolKind::Relax => "Relax",
        }
    }

    pub fn shortcut(self) -> &'static str {
        match self {
            ToolKind::Create => "Q",
            ToolKind::Draw => "D",
            ToolKind::Relax => "R",
        }
    }
}

/// Everything a tool needs to know about the frame.
#[derive(Debug, Clone, Copy)]
pub struct ToolInput<'a> {
    pub surface: Surface<'a>,
    pub view: ViewInfo,
    pub cursor: Vec2,
    pub modifiers: ModifiersState,
}

/// Tool settings shown in the UI.
#[derive(Debug, Clone)]
pub struct ToolSettings {
    /// Width of drawn strips, in pixels at the stroke start.
    pub strip_width: f32,
    /// Relax brush radius, in pixels.
    pub brush_radius: f32,
    /// Relax brush strength per stroke step, 0..1.
    pub brush_strength: f32,
    /// Let relaxing move border vertices (along the border).
    pub relax_borders: bool,
}

impl Default for ToolSettings {
    fn default() -> Self {
        Self {
            strip_width: 40.0,
            brush_radius: 70.0,
            brush_strength: 0.35,
            relax_borders: false,
        }
    }
}

#[derive(Debug)]
pub struct Tools {
    pub active: ToolKind,
    pub settings: ToolSettings,
    /// Message about the last action (e.g. why a face was refused).
    pub status: Option<String>,
    hover_vertex: Option<VertexId>,
    hover_face: Option<FaceId>,
    /// Surface point under the cursor, for previews.
    hover_surface: Option<SurfacePoint>,
    /// Create: vertices of the face being built.
    pending: Vec<VertexId>,
    /// Create: left-button press position, to tell clicks from drags.
    press: Option<(MouseButton, Vec2)>,
    /// Create: vertex being dragged and the gesture key.
    drag: Option<(VertexId, u64)>,
    /// Draw: the stroke being drawn.
    stroke: Vec<SurfacePoint>,
    stroke_cursor: Vec2,
    drawing: bool,
    /// Relax: gesture key while the brush is down.
    brushing: Option<u64>,
}

impl Default for Tools {
    fn default() -> Self {
        Self {
            active: ToolKind::Create,
            settings: ToolSettings::default(),
            status: None,
            hover_vertex: None,
            hover_face: None,
            hover_surface: None,
            pending: Vec::new(),
            press: None,
            drag: None,
            stroke: Vec::new(),
            stroke_cursor: Vec2::ZERO,
            drawing: false,
            brushing: None,
        }
    }
}

impl Tools {
    pub fn select(&mut self, tool: ToolKind) {
        if tool != self.active {
            self.cancel();
            self.active = tool;
        }
    }

    /// Whether a tool gesture is in progress (camera input should wait).
    pub fn is_busy(&self) -> bool {
        self.drag.is_some() || self.drawing || self.brushing.is_some()
    }

    /// Abandons the current gesture or face in progress.
    fn cancel(&mut self) -> bool {
        let had_something = !self.pending.is_empty() || self.drawing;
        self.pending.clear();
        self.stroke.clear();
        self.drawing = false;
        self.press = None;
        had_something
    }

    /// Screen-space brush circle (center, radius) to draw, if any.
    pub fn brush_circle(
        &self,
        input: &ToolInput<'_>,
        cursor_in_viewport: bool,
    ) -> Option<(Vec2, f32)> {
        (self.active == ToolKind::Relax && (cursor_in_viewport || self.brushing.is_some()))
            .then_some((input.cursor, self.settings.brush_radius))
    }

    /// Recomputes what is under the cursor.
    pub fn update_hover(&mut self, retopo: &Retopo, input: &ToolInput<'_>) {
        let mesh = retopo.mesh();
        self.pending.retain(|&v| mesh.contains_vertex(v));
        self.hover_surface = input.surface.raycast(&input.view, input.cursor);
        self.hover_vertex = pick::vertex(
            mesh,
            &input.surface,
            &input.view,
            input.cursor,
            PICK_RADIUS,
            retopo.x_ray,
        );
        self.hover_face = if self.hover_vertex.is_none() {
            pick::face(
                mesh,
                &input.surface,
                &input.view,
                input.cursor,
                retopo.x_ray,
            )
        } else {
            None
        };
    }

    pub fn clear_hover(&mut self) {
        self.hover_vertex = None;
        self.hover_face = None;
        self.hover_surface = None;
    }

    /// Returns true if the tool took the press.
    pub fn mouse_down(
        &mut self,
        retopo: &mut Retopo,
        input: &ToolInput<'_>,
        button: MouseButton,
    ) -> bool {
        self.update_hover(retopo, input);
        match (self.active, button) {
            (ToolKind::Create, MouseButton::Left) if input.modifiers.control_key() => {
                if let Some(v) = self.hover_vertex {
                    self.drag = Some((v, retopo.new_gesture()));
                }
                true
            }
            (ToolKind::Draw, MouseButton::Left) => {
                self.stroke.clear();
                self.drawing = true;
                self.stroke_cursor = input.cursor;
                self.stroke.extend(self.hover_surface);
                true
            }
            (ToolKind::Relax, MouseButton::Left) => {
                let gesture = retopo.new_gesture();
                self.brushing = Some(gesture);
                self.relax_under_brush(retopo, input, gesture);
                true
            }
            (_, MouseButton::Left | MouseButton::Right) => {
                self.press = Some((button, input.cursor));
                true
            }
            _ => false,
        }
    }

    pub fn mouse_move(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>) {
        if let Some((v, gesture)) = self.drag {
            if !retopo.mesh().contains_vertex(v) {
                self.drag = None;
                return;
            }
            if let Some(hit) = input.surface.raycast(&input.view, input.cursor) {
                let symmetry = retopo.symmetry();
                let _ = retopo.doc.edit_merged("Move vertex", gesture, |m| {
                    edit::move_vertex(m, symmetry.as_ref(), v, hit.position);
                    Ok::<_, ()>(())
                });
            }
            return;
        }
        if self.drawing {
            if input.cursor.distance(self.stroke_cursor) >= STROKE_STEP
                && let Some(hit) = input.surface.raycast(&input.view, input.cursor)
            {
                self.stroke.push(hit);
                self.stroke_cursor = input.cursor;
            }
            return;
        }
        if let Some(gesture) = self.brushing {
            self.relax_under_brush(retopo, input, gesture);
            return;
        }
        self.update_hover(retopo, input);
    }

    pub fn mouse_up(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>, button: MouseButton) {
        if button == MouseButton::Left {
            if self.drag.take().is_some() {
                self.update_hover(retopo, input);
                return;
            }
            if self.drawing {
                self.drawing = false;
                self.finish_strip(retopo, input);
                self.update_hover(retopo, input);
                return;
            }
            if self.brushing.take().is_some() {
                return;
            }
        }
        let Some((pressed, at)) = self.press.take() else {
            return;
        };
        if pressed != button || at.distance(input.cursor) > CLICK_SLOP {
            return;
        }
        match button {
            MouseButton::Left if self.active == ToolKind::Create => {
                self.create_click(retopo, input)
            }
            MouseButton::Right => {
                self.cancel();
            }
            _ => {}
        }
    }

    /// Returns true if the tool handled the key.
    pub fn key(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>, key: &Key) -> bool {
        match key {
            Key::Named(NamedKey::Escape) => self.cancel(),
            Key::Named(NamedKey::Delete | NamedKey::Backspace) => {
                self.update_hover(retopo, input);
                let symmetry = retopo.symmetry();
                let result = if let Some(v) = self.hover_vertex {
                    retopo.doc.edit("Delete vertex", |m| {
                        edit::remove_vertex(m, symmetry.as_ref(), v)
                    })
                } else if let Some(f) = self.hover_face {
                    retopo.doc.edit("Delete face", |m| {
                        edit::remove_face(m, symmetry.as_ref(), f)
                    })
                } else {
                    return false;
                };
                self.report(result);
                self.update_hover(retopo, input);
                true
            }
            _ => false,
        }
    }

    // ---- Create ------------------------------------------------------------

    fn create_click(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>) {
        let symmetry = retopo.symmetry();
        let vertex = match self.hover_vertex {
            Some(v) if self.pending.len() >= 3 && self.pending[0] == v => {
                self.commit_face(retopo, input);
                return;
            }
            Some(v) if self.pending.contains(&v) => return,
            Some(v) => v,
            None => match self.hover_surface {
                Some(hit) => {
                    let result = retopo.doc.edit("Add vertex", |m| {
                        Ok::<_, ()>(edit::add_vertex(m, symmetry.as_ref(), hit.position))
                    });
                    match result {
                        Ok(v) => v,
                        Err(()) => return,
                    }
                }
                None => {
                    self.pending.clear();
                    return;
                }
            },
        };
        self.pending.push(vertex);
        if self.pending.len() == 4 {
            self.commit_face(retopo, input);
        }
        self.update_hover(retopo, input);
    }

    fn commit_face(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>) {
        let vertices = std::mem::take(&mut self.pending);
        let mesh = retopo.mesh();
        let centroid =
            vertices.iter().map(|&v| mesh.position(v)).sum::<Vec3>() / vertices.len() as f32;
        let facing = input.view.eye() - centroid;
        let symmetry = retopo.symmetry();
        let result = retopo.doc.edit("Add face", |m| {
            edit::add_face(m, symmetry.as_ref(), &vertices, facing)
        });
        self.report(result.map(|_| ()));
    }

    // ---- Draw --------------------------------------------------------------

    fn finish_strip(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>) {
        let stroke = std::mem::take(&mut self.stroke);
        let Some(first) = stroke.first() else { return };
        let width = input.view.world_per_pixel_at(first.position) * self.settings.strip_width;
        let rows = geometry::strip_rows(&stroke, width, width);
        if rows.len() < 2 {
            self.status = Some("Stroke too short for a strip".into());
            return;
        }
        let symmetry = retopo.symmetry();
        let surface = input.surface;
        let view = input.view;
        let x_ray = retopo.x_ray;
        let mut skipped = 0;
        let result = retopo.doc.edit("Draw strip", |m| {
            // Reuse visible vertices near a strip corner, so strips connect.
            let corner = |m: &mut polyflow_core::halfedge::HalfEdgeMesh, p: Vec3| -> VertexId {
                let p = surface.snap(p);
                let near = view
                    .project(p)
                    .and_then(|px| pick::vertex(m, &surface, &view, px, PICK_RADIUS, x_ray));
                near.unwrap_or_else(|| edit::add_vertex(m, symmetry.as_ref(), p))
            };
            let ids: Vec<[VertexId; 2]> = rows
                .iter()
                .map(|&[l, r]| [corner(m, l), corner(m, r)])
                .collect();
            for (k, pair) in ids.windows(2).enumerate() {
                let [l0, r0] = pair[0];
                let [l1, r1] = pair[1];
                let quad = [r0, r1, l1, l0];
                let unique: std::collections::HashSet<_> = quad.iter().collect();
                if unique.len() < 4 {
                    skipped += 1;
                    continue;
                }
                let normal = stroke[(k * (stroke.len() - 1)) / ids.len().max(1)].normal;
                if edit::add_face(m, symmetry.as_ref(), &quad, normal).is_err() {
                    skipped += 1;
                }
            }
            Ok::<_, ()>(())
        });
        if result.is_ok() {
            self.status =
                (skipped > 0).then(|| format!("{skipped} strip quads could not be added"));
        }
    }

    // ---- Relax -------------------------------------------------------------

    fn relax_under_brush(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>, gesture: u64) {
        let radius = self.settings.brush_radius;
        let strength = self.settings.brush_strength;
        let mesh = retopo.mesh();
        let eye = input.view.eye();
        let mut affected = Vec::new();
        let mut weights = Vec::new();
        for v in mesh.vertices() {
            let p = mesh.position(v);
            let Some(px) = input.view.project(p) else {
                continue;
            };
            let d = px.distance(input.cursor);
            if d <= radius && (retopo.x_ray || input.surface.is_visible(eye, p)) {
                affected.push(v);
                weights.push(strength * geometry::falloff(d / radius));
            }
        }
        if affected.is_empty() {
            return;
        }
        let borders = self.settings.relax_borders;
        relax(
            retopo,
            &input.surface,
            &affected,
            &weights,
            borders,
            Some(gesture),
        );
    }

    // ---- Preview -----------------------------------------------------------

    /// Hover highlights and tool previews.
    pub fn overlay(&self, retopo: &Retopo, out: &mut Overlay) {
        out.clear();
        let mesh = retopo.mesh();
        if let Some(f) = self.hover_face.filter(|&f| mesh.contains_face(f)) {
            let p: Vec<Vec3> = mesh.face_vertices(f).map(|v| mesh.position(v)).collect();
            for i in 1..p.len().saturating_sub(1) {
                for q in [p[0], p[i], p[i + 1]] {
                    out.triangles.push(OverlayVertex {
                        position: q,
                        color: HOVER_FACE_COLOR,
                    });
                }
            }
        }
        let pending: Vec<Vec3> = self
            .pending
            .iter()
            .filter(|&&v| mesh.contains_vertex(v))
            .map(|&v| mesh.position(v))
            .collect();
        for w in pending.windows(2) {
            out.lines.push(OverlayLine {
                a: w[0],
                b: w[1],
                color: PENDING_COLOR,
            });
        }
        if let (Some(last), Some(next)) = (
            pending.last(),
            self.hover_vertex
                .filter(|&v| mesh.contains_vertex(v))
                .map(|v| mesh.position(v))
                .or(self.hover_surface.map(|s| s.position)),
        ) {
            out.lines.push(OverlayLine {
                a: *last,
                b: next,
                color: PENDING_COLOR,
            });
        }
        for &p in &pending {
            out.points.push(OverlayPoint {
                position: p,
                color: PENDING_COLOR,
                size: VERTEX_SIZE + 3.0,
            });
        }
        if let Some(v) = self.hover_vertex.filter(|&v| mesh.contains_vertex(v)) {
            out.points.push(OverlayPoint {
                position: mesh.position(v),
                color: HOVER_COLOR,
                size: VERTEX_SIZE + 5.0,
            });
        }
        for w in self.stroke.windows(2) {
            out.lines.push(OverlayLine {
                a: w[0].position,
                b: w[1].position,
                color: PENDING_COLOR,
            });
        }
    }

    fn report<E: std::fmt::Display>(&mut self, result: Result<(), E>) {
        self.status = result.err().map(|e| e.to_string());
    }
}

/// Relaxes `vertices` by `weights`, snaps them back onto the surface and keeps
/// mirror partners in sync. With a `gesture`, merges into that undo step.
pub fn relax(
    retopo: &mut Retopo,
    surface: &Surface<'_>,
    vertices: &[VertexId],
    weights: &[f32],
    relax_borders: bool,
    gesture: Option<u64>,
) {
    let symmetry = retopo.symmetry();
    let mesh = retopo.mesh();
    let moved: HashMap<VertexId, Vec3> =
        geometry::relaxed_positions(mesh, vertices, weights, relax_borders)
            .into_iter()
            .map(|(v, p)| (v, surface.snap(p)))
            .collect();
    // Pair vertices before moving any: pairing is by position.
    let partners: HashMap<VertexId, VertexId> = match &symmetry {
        Some(sym) => moved
            .keys()
            .filter_map(|&v| sym.partner(mesh, v).map(|w| (v, w)))
            .collect(),
        None => HashMap::new(),
    };
    let apply = |m: &mut polyflow_core::halfedge::HalfEdgeMesh| {
        for (&v, &p) in &moved {
            match (&symmetry, partners.get(&v)) {
                (Some(sym), Some(&w)) if w == v => {
                    let mut q = p;
                    q[sym.axis.index()] = 0.0;
                    m.set_position(v, q);
                }
                // When both partners moved, the one on the positive side leads.
                (Some(sym), Some(&w)) => {
                    if !moved.contains_key(&w) || p[sym.axis.index()] > 0.0 {
                        m.set_position(v, p);
                        m.set_position(w, sym.reflect(p));
                    }
                }
                _ => m.set_position(v, p),
            }
        }
        Ok::<_, ()>(())
    };
    let _ = match gesture {
        Some(key) => retopo.doc.edit_merged("Relax", key, apply),
        None => retopo.doc.edit("Relax", apply),
    };
}

#[cfg(test)]
mod tests {
    use polyflow_core::halfedge::HalfEdgeMesh;
    use polyflow_spatial::Bvh;

    use super::*;
    use crate::retopo::surface::tests::plane;
    use crate::retopo::symmetry::Axis;

    #[test]
    fn relax_keeps_mirror_pairs_exact() {
        let reference = plane(4.0, 8);
        let bvh = Bvh::build(&reference);
        let surface = Surface {
            mesh: &reference,
            bvh: &bvh,
        };
        let mut retopo = Retopo {
            symmetry_axis: Some(Axis::X),
            scene_radius: 4.0,
            ..Default::default()
        };
        let sym = retopo.symmetry().unwrap();
        retopo
            .doc
            .edit("Grid", |m: &mut HalfEdgeMesh| {
                let mut ids = Vec::new();
                for j in 0..3 {
                    for i in 0..5 {
                        let jitter = if (i + j) % 2 == 0 { 0.1 } else { -0.05 };
                        let x = (i as f32 - 2.0) * 0.5;
                        ids.push(edit::add_vertex(
                            m,
                            Some(&sym),
                            Vec3::new(x + jitter * x.signum(), 0.0, j as f32 * 0.5),
                        ));
                    }
                }
                for j in 0..2 {
                    for i in 0..4 {
                        let k = j * 5 + i;
                        let quad = [ids[k], ids[k + 1], ids[k + 6], ids[k + 5]];
                        edit::add_face_facing(m, &quad, Vec3::Y).unwrap();
                    }
                }
                Ok::<_, ()>(())
            })
            .unwrap();
        let all: Vec<VertexId> = retopo.mesh().vertices().collect();
        let weights = vec![0.8; all.len()];
        relax(&mut retopo, &surface, &all, &weights, true, None);
        let mesh = retopo.mesh();
        mesh.validate().unwrap();
        for v in mesh.vertices() {
            let p = mesh.position(v);
            assert!(p.y.abs() < 1e-5, "stays on the plane");
            assert!(sym.partner(mesh, v).is_some(), "{p} lost its partner");
        }
        assert_eq!(retopo.doc.undo_name(), Some("Relax"));
    }
}
