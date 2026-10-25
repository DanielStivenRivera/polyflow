//! Interactive retopology tools.
//!
//! Tools receive mouse and keyboard events that the camera did not take
//! (left and right buttons without Alt) and edit the retopo mesh through
//! `MeshDocument`, so every action is one atomic undo step.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use polyflow_core::glam::{Vec2, Vec3};
use polyflow_core::halfedge::{EdgeId, FaceId, HalfEdgeId, HalfEdgeMesh, TopologyError, VertexId};
use polyflow_render::{Overlay, OverlayLine, OverlayPoint, OverlayVertex};
use winit::event::MouseButton;
use winit::keyboard::{Key, ModifiersState, NamedKey};

use super::pick::Region;
use super::selection::{self, Elements, SelectMode, SelectOp};
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
/// Longest time between the clicks of a double-click.
const DOUBLE_CLICK: Duration = Duration::from_millis(400);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Select,
    Create,
    Draw,
    Relax,
    Loop,
    Cut,
    Extrude,
    Tweak,
}

impl ToolKind {
    pub const ALL: [ToolKind; 8] = [
        ToolKind::Select,
        ToolKind::Create,
        ToolKind::Draw,
        ToolKind::Extrude,
        ToolKind::Tweak,
        ToolKind::Relax,
        ToolKind::Loop,
        ToolKind::Cut,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ToolKind::Select => "Select",
            ToolKind::Create => "Create",
            ToolKind::Draw => "Draw",
            ToolKind::Relax => "Relax",
            ToolKind::Loop => "Loop",
            ToolKind::Cut => "Cut",
            ToolKind::Extrude => "Extrude",
            ToolKind::Tweak => "Tweak",
        }
    }

    pub fn shortcut(self) -> &'static str {
        match self {
            ToolKind::Select => "S",
            ToolKind::Create => "Q",
            ToolKind::Draw => "D",
            ToolKind::Relax => "R",
            ToolKind::Loop => "I",
            ToolKind::Cut => "C",
            ToolKind::Extrude => "E",
            ToolKind::Tweak => "W",
        }
    }
}

/// Edits of the selection, from keys or the retopology panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Delete the selection (Delete), or what is under the cursor.
    Delete,
    /// Dissolve the selection (Backspace).
    Dissolve,
    /// Merge the selected vertices at their center (M).
    Merge,
    /// Split the face between two selected vertices (J).
    Connect,
    /// Rotate the selected edges (T).
    Rotate,
    /// Bridge the two selected borders (B).
    Bridge,
    /// Fill the holes along the selected border edges (H).
    Fill,
    /// Mark the selected edges hard (K).
    MarkHard,
    /// Clear the hard mark of the selected edges (Shift+K).
    ClearHard,
}

/// An extrusion being dragged out.
#[derive(Debug, Clone)]
struct Extrusion {
    halfedges: Vec<HalfEdgeId>,
    /// Unit direction, along the surface and away from the faces, per vertex.
    directions: HashMap<VertexId, Vec3>,
    /// Surface point where the drag started, and the direction measured along.
    start: Vec3,
    axis: Vec3,
    distance: f32,
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
    /// Select: drag a lasso instead of a box.
    pub lasso: bool,
    hover_vertex: Option<VertexId>,
    hover_edge: Option<EdgeId>,
    /// Where along `hover_edge`'s first half-edge the cursor is.
    hover_edge_t: f32,
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
    /// Select: the area being dragged.
    region: Option<Region>,
    /// Select: time and place of the last click, for double-clicks.
    last_click: Option<(Instant, Vec2)>,
    /// Loop: the edge being slid along while the button is down.
    sliding: Option<HalfEdgeId>,
    /// Loop: the half-edge and fraction the loop would be inserted at.
    loop_at: Option<(HalfEdgeId, f32)>,
    /// Loop: insert in the middle of the ring's edges (Shift held).
    even: bool,
    /// Cut: the vertex the next cut starts from.
    cut_from: Option<VertexId>,
    /// Extrude: the extrusion being dragged.
    extrusion: Option<Extrusion>,
    /// Tweak: the brush stroke in progress.
    tweak: Option<Tweak>,
}

/// A Tweak brush stroke.
#[derive(Debug, Clone)]
struct Tweak {
    /// Vertices under the brush when it went down, with their positions then
    /// and their weights.
    vertices: Vec<(VertexId, Vec3, f32)>,
    /// The surface point the brush went down on.
    start: Vec3,
    gesture: u64,
}

impl Default for Tools {
    fn default() -> Self {
        Self {
            active: ToolKind::Create,
            settings: ToolSettings::default(),
            status: None,
            lasso: false,
            hover_vertex: None,
            hover_edge: None,
            hover_edge_t: 0.5,
            hover_face: None,
            hover_surface: None,
            pending: Vec::new(),
            press: None,
            drag: None,
            stroke: Vec::new(),
            stroke_cursor: Vec2::ZERO,
            drawing: false,
            brushing: None,
            region: None,
            last_click: None,
            sliding: None,
            loop_at: None,
            even: false,
            cut_from: None,
            extrusion: None,
            tweak: None,
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

    /// Forgets all per-mesh state, for when the mesh is replaced.
    pub fn reset(&mut self) {
        self.cancel();
        self.drag = None;
        self.brushing = None;
        self.clear_hover();
    }

    /// Whether a tool gesture is in progress (camera input should wait).
    pub fn is_busy(&self) -> bool {
        self.drag.is_some()
            || self.drawing
            || self.brushing.is_some()
            || self.region.is_some()
            || self.sliding.is_some()
            || self.extrusion.is_some()
            || self.tweak.is_some()
    }

    /// Abandons the current gesture or face in progress.
    fn cancel(&mut self) -> bool {
        let had_something = !self.pending.is_empty()
            || self.drawing
            || self.region.is_some()
            || self.sliding.is_some()
            || self.cut_from.is_some()
            || self.extrusion.is_some();
        self.pending.clear();
        self.sliding = None;
        self.cut_from = None;
        self.extrusion = None;
        self.stroke.clear();
        self.drawing = false;
        self.region = None;
        self.press = None;
        had_something
    }

    /// Screen-space brush circle (center, radius) to draw, if any.
    pub fn brush_circle(
        &self,
        input: &ToolInput<'_>,
        cursor_in_viewport: bool,
    ) -> Option<(Vec2, f32)> {
        let brush = matches!(self.active, ToolKind::Relax | ToolKind::Tweak);
        let down = self.brushing.is_some() || self.tweak.is_some();
        (brush && (cursor_in_viewport || down))
            .then_some((input.cursor, self.settings.brush_radius))
    }

    /// The selection area being dragged, as a closed outline in window
    /// pixels.
    pub fn region_outline(&self) -> Option<Vec<Vec2>> {
        match self.region.as_ref()? {
            Region::Box(a, b) => Some(vec![*a, Vec2::new(b.x, a.y), *b, Vec2::new(a.x, b.y)]),
            Region::Lasso(points) => Some(points.clone()),
        }
    }

    /// Recomputes what is under the cursor.
    pub fn update_hover(&mut self, retopo: &Retopo, input: &ToolInput<'_>) {
        let mesh = retopo.mesh();
        self.even = input.modifiers.shift_key();
        self.pending.retain(|&v| mesh.contains_vertex(v));
        self.hover_surface = input.surface.raycast(&input.view, input.cursor);
        // What each tool works on; the Select tool, on the selection mode.
        let (vertices, edges, faces) = match self.active {
            ToolKind::Select => match retopo.selection.mode() {
                SelectMode::Vertex => (true, false, false),
                SelectMode::Edge => (false, true, false),
                SelectMode::Face => (false, false, true),
            },
            ToolKind::Loop | ToolKind::Extrude => (false, true, false),
            ToolKind::Cut => (true, true, false),
            _ => (true, false, true),
        };
        self.hover_vertex = if vertices {
            pick::vertex(
                mesh,
                &input.surface,
                &input.view,
                input.cursor,
                PICK_RADIUS,
                retopo.x_ray,
            )
        } else {
            None
        };
        let borders_only = self.active == ToolKind::Extrude;
        self.hover_edge = if edges && self.hover_vertex.is_none() {
            pick::edge_where(
                mesh,
                &input.surface,
                &input.view,
                input.cursor,
                PICK_RADIUS,
                retopo.x_ray,
                |e| !borders_only || mesh.is_boundary_edge(e),
            )
        } else {
            None
        };
        if let Some(e) = self.hover_edge {
            self.hover_edge_t =
                pick::halfedge_param(mesh, &input.view, e.halfedge(0), input.cursor);
        }
        if self.active == ToolKind::Loop && self.sliding.is_none() {
            self.loop_at = self.hover_edge.map(|e| (e.halfedge(0), self.hover_edge_t));
        }
        self.hover_face = if self.hover_vertex.is_none() && faces {
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
        self.hover_edge = None;
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
            (ToolKind::Loop, MouseButton::Left) => {
                self.sliding = self.loop_at.map(|(h, _)| h);
                true
            }
            (ToolKind::Extrude, MouseButton::Left) => {
                self.start_extrusion(retopo, input);
                true
            }
            (ToolKind::Tweak, MouseButton::Left) => {
                self.start_tweak(retopo, input);
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
        if let Some(extrusion) = &mut self.extrusion {
            if let Some(hit) = input.surface.raycast(&input.view, input.cursor) {
                extrusion.distance = (hit.position - extrusion.start)
                    .dot(extrusion.axis)
                    .max(0.0);
            }
            return;
        }
        if self.tweak.is_some() {
            self.tweak_under_brush(retopo, input);
            return;
        }
        if let Some(h) = self.sliding {
            if !retopo.mesh().contains_halfedge(h) {
                self.sliding = None;
                return;
            }
            self.even = input.modifiers.shift_key();
            let t = pick::halfedge_param(retopo.mesh(), &input.view, h, input.cursor);
            self.loop_at = Some((h, t));
            return;
        }
        if self.active == ToolKind::Select {
            match (&mut self.region, self.press) {
                (Some(Region::Box(_, end)), _) => *end = input.cursor,
                (Some(Region::Lasso(points)), _) => {
                    if points
                        .last()
                        .is_none_or(|p| p.distance(input.cursor) >= STROKE_STEP)
                    {
                        points.push(input.cursor);
                    }
                }
                (None, Some((MouseButton::Left, at))) if at.distance(input.cursor) > CLICK_SLOP => {
                    self.region = Some(if self.lasso {
                        Region::Lasso(vec![at, input.cursor])
                    } else {
                        Region::Box(at, input.cursor)
                    });
                    self.clear_hover();
                }
                _ => {}
            }
            if self.region.is_some() {
                return;
            }
        }
        if let Some((v, gesture)) = self.drag {
            if !retopo.mesh().contains_vertex(v) {
                self.drag = None;
                return;
            }
            if let Some(hit) = input.surface.raycast(&input.view, input.cursor) {
                let symmetry = retopo.symmetry();
                let _ = retopo.edit_merged("Move vertex", gesture, |m| {
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
            if let Some((v, gesture)) = self.drag.take() {
                self.weld(retopo, input, v, gesture);
                self.update_hover(retopo, input);
                return;
            }
            if self.sliding.take().is_some() {
                self.insert_loop(retopo, input);
                self.update_hover(retopo, input);
                return;
            }
            if let Some(extrusion) = self.extrusion.take() {
                self.finish_extrusion(retopo, input, extrusion);
                self.update_hover(retopo, input);
                return;
            }
            if self.tweak.take().is_some() {
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
            if let Some(region) = self.region.take() {
                self.press = None;
                let elements = pick::in_region(
                    retopo.mesh(),
                    &input.surface,
                    &input.view,
                    &region,
                    retopo.selection.mode(),
                    retopo.x_ray,
                );
                retopo.selection.apply(select_op(input.modifiers), elements);
                self.update_hover(retopo, input);
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
            MouseButton::Left if self.active == ToolKind::Select => {
                self.select_click(retopo, input)
            }
            MouseButton::Left if self.active == ToolKind::Cut => self.cut_click(retopo, input),
            MouseButton::Right => {
                self.cancel();
            }
            _ => {}
        }
    }

    /// Returns true if the tool handled the key.
    pub fn key(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>, key: &Key) -> bool {
        let character = match key {
            Key::Character(c) => Some(c.to_ascii_lowercase()),
            _ => None,
        };
        let mode = match character.as_deref() {
            Some("1") => Some(SelectMode::Vertex),
            Some("2") => Some(SelectMode::Edge),
            Some("3") => Some(SelectMode::Face),
            _ => None,
        };
        if let Some(mode) = mode {
            self.set_select_mode(retopo, input, mode);
            return true;
        }
        match character.as_deref() {
            Some("a") => {
                if retopo.selection.is_empty() {
                    retopo.selection.select_all(retopo.doc.mesh());
                } else {
                    retopo.selection.clear();
                }
                return true;
            }
            Some("l") => {
                self.grow_loops(retopo, input.modifiers.shift_key());
                return true;
            }
            Some("m") => {
                self.run(retopo, input, Command::Merge);
                return true;
            }
            Some("j") => {
                self.run(retopo, input, Command::Connect);
                return true;
            }
            Some("t") => {
                self.run(retopo, input, Command::Rotate);
                return true;
            }
            Some("b") if !input.modifiers.control_key() => {
                self.run(retopo, input, Command::Bridge);
                return true;
            }
            Some("h") => {
                self.run(retopo, input, Command::Fill);
                return true;
            }
            Some("k") => {
                let command = if input.modifiers.shift_key() {
                    Command::ClearHard
                } else {
                    Command::MarkHard
                };
                self.run(retopo, input, command);
                return true;
            }
            _ => {}
        }
        match key {
            Key::Named(NamedKey::Escape) => {
                if !self.cancel() {
                    retopo.selection.clear();
                }
                true
            }
            Key::Named(NamedKey::Delete) => self.run(retopo, input, Command::Delete),
            Key::Named(NamedKey::Backspace) => self.run(retopo, input, Command::Dissolve),
            _ => false,
        }
    }

    /// Runs an edit command on the selection. Returns false if there was
    /// nothing to do it on.
    pub fn run(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>, command: Command) -> bool {
        self.update_hover(retopo, input);
        let symmetry = retopo.symmetry();
        let sym = symmetry.as_ref();
        let selection = retopo.selection.clone();
        let mode = selection.mode();
        let result: Result<(), TopologyError> = match command {
            Command::Delete if !selection.is_empty() => {
                retopo.edit("Delete", |m| edit::delete_selection(m, sym, &selection))
            }
            Command::Delete => {
                if let Some(v) = self.hover_vertex {
                    retopo.edit("Delete vertex", |m| edit::remove_vertex(m, sym, v))
                } else if let Some(f) = self.hover_face {
                    retopo.edit("Delete face", |m| edit::remove_face(m, sym, f))
                } else {
                    return false;
                }
            }
            Command::Dissolve if selection.is_empty() => {
                self.status = Some("Select vertices, edges or faces to dissolve".into());
                return false;
            }
            Command::Dissolve => {
                retopo.edit("Dissolve", |m| edit::dissolve_selection(m, sym, &selection))
            }
            Command::Merge => {
                let vertices = selected_vertices(retopo.mesh(), &selection);
                if vertices.len() < 2 {
                    self.status = Some("Select two or more vertices to merge".into());
                    return false;
                }
                let mesh = retopo.mesh();
                let center = vertices.iter().map(|&v| mesh.position(v)).sum::<Vec3>()
                    / vertices.len() as f32;
                let position = input.surface.snap(center);
                let result = retopo.edit("Merge vertices", |m| {
                    edit::merge_vertices(m, sym, &vertices, position)
                });
                if let Ok(kept) = result
                    && mode == SelectMode::Vertex
                {
                    retopo
                        .selection
                        .apply(SelectOp::Replace, Elements::Vertices(vec![kept]));
                }
                result.map(|_| ())
            }
            Command::Connect => {
                let vertices = selected_vertices(retopo.mesh(), &selection);
                let [a, b] = vertices[..] else {
                    self.status = Some("Select the two vertices to connect".into());
                    return false;
                };
                let result = retopo.edit("Connect vertices", |m| edit::connect(m, sym, a, b));
                if let Ok(e) = result {
                    retopo
                        .selection
                        .set_mode(retopo.doc.mesh(), SelectMode::Edge);
                    retopo
                        .selection
                        .apply(SelectOp::Replace, Elements::Edges(vec![e]));
                }
                result.map(|_| ())
            }
            Command::Rotate => {
                let edges: Vec<EdgeId> = match mode {
                    SelectMode::Edge => selection.edges().iter().copied().collect(),
                    _ => self.hover_edge.into_iter().collect(),
                };
                if edges.is_empty() {
                    self.status = Some("Select the edges to rotate".into());
                    return false;
                }
                let result = retopo.edit("Rotate edges", |m| edit::rotate_edges(m, sym, &edges));
                if let Ok(turned) = &result
                    && mode == SelectMode::Edge
                {
                    retopo
                        .selection
                        .apply(SelectOp::Replace, Elements::Edges(turned.clone()));
                }
                result.map(|_| ())
            }
            Command::Bridge | Command::Fill => {
                let mesh = retopo.mesh();
                let mut edges: Vec<EdgeId> = selection
                    .edges_for_loops(mesh)
                    .into_iter()
                    .filter(|&e| mesh.is_boundary_edge(e))
                    .collect();
                if edges.is_empty() && command == Command::Fill {
                    edges.extend(self.hover_edge.filter(|&e| mesh.is_boundary_edge(e)));
                }
                if edges.is_empty() {
                    self.status = Some(match command {
                        Command::Bridge => "Select the edges of two borders to bridge".into(),
                        _ => "Select an edge on the border of the hole to fill".into(),
                    });
                    return false;
                }
                if command == Command::Bridge {
                    retopo.edit("Bridge", |m| edit::bridge(m, sym, &edges))
                } else {
                    let surface = input.surface;
                    retopo.edit("Fill hole", |m| {
                        let inner = edit::fill_holes(m, sym, &edges)?;
                        snap_vertices(m, sym, &surface, &inner);
                        Ok(())
                    })
                }
            }
            Command::MarkHard | Command::ClearHard => {
                let hard = command == Command::MarkHard;
                let mut edges = edit::hard_edge_candidates(retopo.mesh(), &selection);
                if edges.is_empty() {
                    edges.extend(self.hover_edge);
                }
                if edges.is_empty() {
                    self.status = Some("Select the edges to mark".into());
                    return false;
                }
                let name = if hard { "Mark hard" } else { "Mark smooth" };
                retopo.edit(name, |m| {
                    edit::set_hard(m, sym, &edges, hard);
                    Ok(())
                })
            }
        };
        self.report(result);
        self.update_hover(retopo, input);
        true
    }

    // ---- Select ------------------------------------------------------------

    pub fn set_select_mode(
        &mut self,
        retopo: &mut Retopo,
        input: &ToolInput<'_>,
        mode: SelectMode,
    ) {
        retopo.selection.set_mode(retopo.doc.mesh(), mode);
        self.update_hover(retopo, input);
    }

    /// Adds the edge loops (or rings) through the selection.
    pub fn grow_loops(&mut self, retopo: &mut Retopo, ring: bool) {
        let mesh = retopo.doc.mesh();
        if ring && retopo.selection.mode() == SelectMode::Face {
            self.status = Some("Rings are made of edges: select edges or vertices".into());
            return;
        }
        let edges = retopo.selection.edges_for_loops(mesh);
        if edges.is_empty() {
            self.status = Some(match retopo.selection.mode() {
                SelectMode::Face => "Select two neighboring faces to select their loop".into(),
                _ => "Select an edge (or two neighboring vertices) first".into(),
            });
            return;
        }
        retopo.selection.grow_loops(mesh, &edges, ring);
    }

    fn select_click(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>) {
        self.update_hover(retopo, input);
        let now = Instant::now();
        let double = self.last_click.take().is_some_and(|(at, px)| {
            now.duration_since(at) <= DOUBLE_CLICK && px.distance(input.cursor) <= CLICK_SLOP
        });
        let op = select_op(input.modifiers);
        let mode = retopo.selection.mode();
        if double {
            // The loop through the edge under the cursor.
            if let Some(e) = self.edge_near_cursor(retopo, input) {
                let elements = selection::loop_elements(retopo.mesh(), mode, e, false);
                retopo.selection.apply(op, elements);
            }
            return;
        }
        self.last_click = Some((now, input.cursor));
        let picked = match mode {
            SelectMode::Vertex => self.hover_vertex.map(|v| Elements::Vertices(vec![v])),
            SelectMode::Edge => self.hover_edge.map(|e| Elements::Edges(vec![e])),
            SelectMode::Face => self.hover_face.map(|f| Elements::Faces(vec![f])),
        };
        match picked {
            Some(elements) => retopo.selection.apply(op, elements),
            None if op == SelectOp::Replace => retopo.selection.clear(),
            None => {}
        }
    }

    /// The edge a double-click means: in face mode, the side of the face
    /// under the cursor nearest to it; otherwise the edge under the cursor.
    fn edge_near_cursor(&self, retopo: &Retopo, input: &ToolInput<'_>) -> Option<EdgeId> {
        let mesh = retopo.mesh();
        if let Some(f) = self.hover_face.filter(|&f| mesh.contains_face(f)) {
            let distance = |h| {
                let a = input.view.project(mesh.position(mesh.origin(h)))?;
                let b = input.view.project(mesh.position(mesh.target(h)))?;
                Some(pick::distance_to_segment(input.cursor, a, b))
            };
            return mesh
                .face_halfedges(f)
                .filter_map(|h| Some((distance(h)?, h.edge())))
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, e)| e);
        }
        self.hover_edge.or_else(|| {
            pick::edge(
                mesh,
                &input.surface,
                &input.view,
                input.cursor,
                PICK_RADIUS,
                retopo.x_ray,
            )
        })
    }

    // ---- Extrude -----------------------------------------------------------

    /// Starts dragging out the border edges under the cursor: the selected
    /// border edges if the hovered one is among them, the hovered edge's
    /// whole border with Shift, else just the hovered edge.
    fn start_extrusion(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>) {
        let mesh = retopo.mesh();
        let Some(hovered) = self.hover_edge.filter(|&e| mesh.is_boundary_edge(e)) else {
            return;
        };
        let Some(hit) = input.surface.raycast(&input.view, input.cursor) else {
            return;
        };
        let selection = &retopo.selection;
        let mut edges: Vec<EdgeId> =
            if selection.mode() == SelectMode::Edge && selection.edges().contains(&hovered) {
                selection
                    .edges()
                    .iter()
                    .copied()
                    .filter(|&e| mesh.is_boundary_edge(e))
                    .collect()
            } else if input.modifiers.shift_key() {
                mesh.edge_loop(hovered)
            } else {
                vec![hovered]
            };
        let symmetry = retopo.symmetry();
        edges = edit::edges_with_mirrors(mesh, symmetry.as_ref(), &edges);
        let halfedges: Vec<HalfEdgeId> = mesh.border_chains(&edges).concat();
        let directions = border_directions(mesh, &input.surface, &halfedges);
        let border = [hovered.halfedge(0), hovered.halfedge(1)]
            .into_iter()
            .find(|&h| mesh.is_boundary_halfedge(h))
            .expect("a border edge");
        let axis = [mesh.origin(border), mesh.target(border)]
            .iter()
            .filter_map(|v| directions.get(v))
            .sum::<Vec3>()
            .normalize_or_zero();
        self.extrusion = Some(Extrusion {
            halfedges,
            directions,
            start: hit.position,
            axis,
            distance: 0.0,
        });
        self.clear_hover();
    }

    /// Where each vertex of the extrusion goes.
    fn extruded_positions(
        mesh: &HalfEdgeMesh,
        surface: &Surface<'_>,
        extrusion: &Extrusion,
    ) -> HashMap<VertexId, Vec3> {
        extrusion
            .directions
            .iter()
            .map(|(&v, &d)| (v, surface.snap(mesh.position(v) + d * extrusion.distance)))
            .collect()
    }

    fn finish_extrusion(
        &mut self,
        retopo: &mut Retopo,
        input: &ToolInput<'_>,
        extrusion: Extrusion,
    ) {
        let mesh = retopo.mesh();
        if extrusion.distance <= 0.0 || extrusion.halfedges.is_empty() {
            return;
        }
        let positions = Self::extruded_positions(mesh, &input.surface, &extrusion);
        let symmetry = retopo.symmetry();
        let halfedges = extrusion.halfedges.clone();
        let result = retopo.edit("Extrude", |m| {
            edit::extrude(m, symmetry.as_ref(), &halfedges, &positions)
        });
        match result {
            Ok(copies) => {
                // Select the new border, ready to extrude again.
                let mesh = retopo.doc.mesh();
                let edges: Vec<EdgeId> = halfedges
                    .iter()
                    .filter_map(|&h| {
                        let (a, b) = (copies.get(&mesh.origin(h))?, copies.get(&mesh.target(h))?);
                        mesh.find_halfedge(*a, *b).map(|h| h.edge())
                    })
                    .collect();
                retopo.selection.set_mode(mesh, SelectMode::Edge);
                retopo
                    .selection
                    .apply(SelectOp::Replace, Elements::Edges(edges));
                self.status = None;
            }
            Err(e) => self.status = Some(e.to_string()),
        }
    }

    // ---- Tweak -------------------------------------------------------------

    fn start_tweak(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>) {
        let Some(hit) = input.surface.raycast(&input.view, input.cursor) else {
            return;
        };
        let radius = self.settings.brush_radius;
        let mesh = retopo.mesh();
        let eye = input.view.eye();
        let affected: Vec<(VertexId, Vec3, f32)> = mesh
            .vertices()
            .filter_map(|v| {
                let p = mesh.position(v);
                let d = input.view.project(p)?.distance(input.cursor);
                (d <= radius && (retopo.x_ray || input.surface.is_visible(eye, p)))
                    .then(|| (v, p, geometry::falloff(d / radius)))
            })
            .collect();
        if !affected.is_empty() {
            self.tweak = Some(Tweak {
                vertices: affected,
                start: hit.position,
                gesture: retopo.new_gesture(),
            });
        }
    }

    /// Moves the vertices under the brush with the cursor, by their weights,
    /// keeping them on the surface.
    fn tweak_under_brush(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>) {
        let Some(tweak) = &self.tweak else {
            return;
        };
        let Some(hit) = input.surface.raycast(&input.view, input.cursor) else {
            return;
        };
        let delta = hit.position - tweak.start;
        let mesh = retopo.mesh();
        let moved: HashMap<VertexId, Vec3> = tweak
            .vertices
            .iter()
            .filter(|(v, ..)| mesh.contains_vertex(*v))
            .map(|&(v, p, w)| (v, input.surface.snap(p + delta * w)))
            .collect();
        let gesture = tweak.gesture;
        move_vertices(retopo, &moved, "Tweak", Some(gesture));
    }

    // ---- Loop --------------------------------------------------------------

    /// The fraction to insert the loop at, given the cursor's.
    fn loop_t(&self, t: f32) -> f32 {
        if self.even { 0.5 } else { t.clamp(0.02, 0.98) }
    }

    fn insert_loop(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>) {
        let Some((h, t)) = self.loop_at.take() else {
            return;
        };
        let t = self.loop_t(t);
        let symmetry = retopo.symmetry();
        let surface = input.surface;
        let result = retopo.edit("Insert edge loop", |m| {
            let new = edit::insert_loop(m, symmetry.as_ref(), h, t)?;
            snap_vertices(m, symmetry.as_ref(), &surface, &new);
            Ok::<_, TopologyError>(())
        });
        self.report(result);
    }

    /// Positions of the loop the tool would insert: one polyline per ring.
    fn loop_preview(&self, retopo: &Retopo) -> Vec<(Vec<Vec3>, bool)> {
        let mesh = retopo.mesh();
        let Some((h, t)) = self.loop_at.filter(|(h, _)| mesh.contains_halfedge(*h)) else {
            return Vec::new();
        };
        let symmetry = retopo.symmetry();
        let (t, mirror) = edit::loop_plan(mesh, symmetry.as_ref(), h, self.loop_t(t));
        std::iter::once(h)
            .chain(mirror)
            .map(|h| {
                let (ring, closed) = mesh.ring_halfedges(h);
                let points = ring
                    .iter()
                    .map(|&r| {
                        mesh.position(mesh.origin(r))
                            .lerp(mesh.position(mesh.target(r)), t)
                    })
                    .collect();
                (points, closed)
            })
            .collect()
    }

    // ---- Cut ---------------------------------------------------------------

    /// Splits the edge under the cursor (or takes the vertex there) and joins
    /// it to the previous cut point across the face they share.
    fn cut_click(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>) {
        let target = match (self.hover_vertex, self.hover_edge) {
            (Some(v), _) => Ok(v),
            (None, Some(e)) => Err((e.halfedge(0), self.hover_edge_t.clamp(0.0, 1.0))),
            (None, None) => {
                self.cut_from = None;
                return;
            }
        };
        let from = self.cut_from.filter(|&v| retopo.mesh().contains_vertex(v));
        let symmetry = retopo.symmetry();
        let sym = symmetry.as_ref();
        let surface = input.surface;
        let mut joined = false;
        let result = retopo.edit("Cut", |m| {
            let v = match target {
                Ok(v) => v,
                Err((h, t)) => {
                    let v = edit::split_edge(m, sym, h, t)?;
                    snap_vertices(m, sym, &surface, &[v]);
                    v
                }
            };
            if let Some(from) = from.filter(|&f| f != v) {
                joined = edit::connect(m, sym, from, v).is_ok();
            }
            Ok::<_, TopologyError>(v)
        });
        match result {
            Ok(v) => {
                self.status = (from.is_some() && !joined && target.is_err())
                    .then(|| "The cut points do not share a face; starting a new cut".into());
                self.cut_from = Some(v);
            }
            Err(e) => self.status = Some(e.to_string()),
        }
        self.update_hover(retopo, input);
    }

    /// Where the next cut point would go.
    fn cut_target(&self, mesh: &HalfEdgeMesh) -> Option<Vec3> {
        if let Some(v) = self.hover_vertex.filter(|&v| mesh.contains_vertex(v)) {
            return Some(mesh.position(v));
        }
        let h = self.hover_edge?.halfedge(0);
        mesh.contains_halfedge(h).then(|| {
            mesh.position(mesh.origin(h))
                .lerp(mesh.position(mesh.target(h)), self.hover_edge_t)
        })
    }

    // ---- Create ------------------------------------------------------------

    /// After dragging `v` onto another vertex, merges them into that one.
    fn weld(&mut self, retopo: &mut Retopo, input: &ToolInput<'_>, v: VertexId, gesture: u64) {
        let mesh = retopo.mesh();
        if !mesh.contains_vertex(v) {
            return;
        }
        let Some(target) = pick::vertex_except(
            mesh,
            &input.surface,
            &input.view,
            input.cursor,
            PICK_RADIUS,
            retopo.x_ray,
            Some(v),
        ) else {
            return;
        };
        let position = mesh.position(target);
        let symmetry = retopo.symmetry();
        let result = retopo.edit_merged("Weld vertices", gesture, |m| {
            edit::merge_vertices(m, symmetry.as_ref(), &[target, v], position)
        });
        self.report(result.map(|_| ()));
    }

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
                    let result = retopo.edit("Add vertex", |m| {
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
        let result = retopo.edit("Add face", |m| {
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
        let result = retopo.edit("Draw strip", |m| {
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
                    out.triangles.push(OverlayVertex::flat(q, HOVER_FACE_COLOR));
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
        if let Some(e) = self
            .hover_edge
            .filter(|&e| mesh.contains_halfedge(e.halfedge(0)))
        {
            let h = e.halfedge(0);
            out.lines.push(OverlayLine {
                a: mesh.position(mesh.origin(h)),
                b: mesh.position(mesh.target(h)),
                color: HOVER_COLOR,
            });
        }
        if let Some(v) = self.hover_vertex.filter(|&v| mesh.contains_vertex(v)) {
            out.points.push(OverlayPoint {
                position: mesh.position(v),
                color: HOVER_COLOR,
                size: VERTEX_SIZE + 5.0,
            });
        }
        if let Some(extrusion) = &self.extrusion
            && extrusion.distance > 0.0
        {
            for &h in &extrusion.halfedges {
                if !mesh.contains_halfedge(h) {
                    continue;
                }
                let (a, b) = (mesh.origin(h), mesh.target(h));
                let to = |v: VertexId| {
                    mesh.position(v)
                        + extrusion.directions.get(&v).copied().unwrap_or_default()
                            * extrusion.distance
                };
                for (p, q) in [
                    (mesh.position(a), to(a)),
                    (to(a), to(b)),
                    (to(b), mesh.position(b)),
                ] {
                    out.lines.push(OverlayLine {
                        a: p,
                        b: q,
                        color: PENDING_COLOR,
                    });
                }
            }
        }
        for (points, closed) in self.loop_preview(retopo) {
            let n = points.len();
            let segments = if closed { n } else { n.saturating_sub(1) };
            for i in 0..segments {
                out.lines.push(OverlayLine {
                    a: points[i],
                    b: points[(i + 1) % n],
                    color: PENDING_COLOR,
                });
            }
            for p in points {
                out.points.push(OverlayPoint {
                    position: p,
                    color: PENDING_COLOR,
                    size: VERTEX_SIZE,
                });
            }
        }
        if self.active == ToolKind::Cut
            && let Some(target) = self.cut_target(mesh)
        {
            if let Some(from) = self.cut_from.filter(|&v| mesh.contains_vertex(v)) {
                out.lines.push(OverlayLine {
                    a: mesh.position(from),
                    b: target,
                    color: PENDING_COLOR,
                });
            }
            out.points.push(OverlayPoint {
                position: target,
                color: HOVER_COLOR,
                size: VERTEX_SIZE + 3.0,
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

/// The selected vertices, or the vertices of the selected edges or faces.
fn selected_vertices(mesh: &HalfEdgeMesh, selection: &selection::Selection) -> Vec<VertexId> {
    let mut vertices: Vec<VertexId> = match selection.mode() {
        SelectMode::Vertex => selection.vertices().iter().copied().collect(),
        SelectMode::Edge => selection
            .edges()
            .iter()
            .flat_map(|e| [mesh.origin(e.halfedge(0)), mesh.origin(e.halfedge(1))])
            .collect(),
        SelectMode::Face => selection
            .faces()
            .iter()
            .flat_map(|&f| mesh.face_vertices(f).collect::<Vec<_>>())
            .collect(),
    };
    vertices.sort_unstable();
    vertices.dedup();
    vertices
}

/// For each vertex on border half-edges: the unit direction along the
/// surface, across the border and away from its faces.
fn border_directions(
    mesh: &HalfEdgeMesh,
    surface: &Surface<'_>,
    halfedges: &[HalfEdgeId],
) -> HashMap<VertexId, Vec3> {
    let mut sums: HashMap<VertexId, Vec3> = HashMap::new();
    for &h in halfedges {
        let (a, b) = (mesh.position(mesh.origin(h)), mesh.position(mesh.target(h)));
        let mid = (a + b) * 0.5;
        let normal = surface.closest(mid).map_or(Vec3::Y, |s| s.normal);
        let mut out = normal.cross(b - a).normalize_or_zero();
        if let Some(f) = mesh.face_of(h.twin()) {
            let n = mesh.face_vertices(f).count() as f32;
            let center = mesh
                .face_vertices(f)
                .map(|v| mesh.position(v))
                .sum::<Vec3>()
                / n;
            if out.dot(center - mid) > 0.0 {
                out = -out;
            }
        }
        // Longer edges weigh more, so short ones do not skew corners.
        let weight = (b - a).length();
        for v in [mesh.origin(h), mesh.target(h)] {
            *sums.entry(v).or_default() += out * weight;
        }
    }
    sums.into_iter()
        .map(|(v, d)| (v, d.normalize_or_zero()))
        .collect()
}

/// Moves new vertices onto the surface, with their mirror partners.
fn snap_vertices(
    mesh: &mut HalfEdgeMesh,
    symmetry: Option<&super::symmetry::Symmetry>,
    surface: &Surface<'_>,
    vertices: &[VertexId],
) {
    for &v in vertices {
        if mesh.contains_vertex(v) {
            let p = surface.snap(mesh.position(v));
            edit::move_vertex(mesh, symmetry, v, p);
        }
    }
}

/// Shift adds to the selection, Ctrl removes from it.
fn select_op(modifiers: ModifiersState) -> SelectOp {
    if modifiers.shift_key() {
        SelectOp::Add
    } else if modifiers.control_key() {
        SelectOp::Remove
    } else {
        SelectOp::Replace
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
    let moved: HashMap<VertexId, Vec3> =
        geometry::relaxed_positions(retopo.mesh(), vertices, weights, relax_borders)
            .into_iter()
            .map(|(v, p)| (v, surface.snap(p)))
            .collect();
    move_vertices(retopo, &moved, "Relax", gesture);
}

/// Moves vertices and keeps mirror partners in sync: seam vertices stay on
/// the seam, and when both partners move, the one on the positive side
/// leads. With a `gesture`, merges into that undo step.
pub fn move_vertices(
    retopo: &mut Retopo,
    moved: &HashMap<VertexId, Vec3>,
    name: &str,
    gesture: Option<u64>,
) {
    let symmetry = retopo.symmetry();
    let mesh = retopo.mesh();
    // Pair vertices before moving any: pairing is by position.
    let partners: HashMap<VertexId, VertexId> = match &symmetry {
        Some(sym) => moved
            .keys()
            .filter_map(|&v| sym.partner(mesh, v).map(|w| (v, w)))
            .collect(),
        None => HashMap::new(),
    };
    let apply = |m: &mut HalfEdgeMesh| {
        for (&v, &p) in moved {
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
        Some(key) => retopo.edit_merged(name, key, apply),
        None => retopo.edit(name, apply),
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

    /// A 3x3 grid of unit quads on the XZ plane, centered on the origin.
    fn grid_retopo() -> Retopo {
        let mut retopo = Retopo::default();
        retopo
            .edit("Grid", |m: &mut HalfEdgeMesh| {
                let ids: Vec<VertexId> = (0..16)
                    .map(|k| {
                        let (i, j) = ((k % 4) as f32, (k / 4) as f32);
                        m.add_vertex(Vec3::new(i - 1.5, 0.0, j - 1.5))
                    })
                    .collect();
                for j in 0..3 {
                    for i in 0..3 {
                        let k = j * 4 + i;
                        let quad = [ids[k], ids[k + 1], ids[k + 5], ids[k + 4]];
                        edit::add_face_facing(m, &quad, Vec3::Y).unwrap();
                    }
                }
                Ok::<_, ()>(())
            })
            .unwrap();
        retopo
    }

    fn top_view() -> ViewInfo {
        ViewInfo {
            camera: polyflow_core::camera::OrbitCamera {
                pitch: 1.2,
                distance: 8.0,
                ..Default::default()
            },
            viewport: polyflow_render::Viewport {
                x: 0.0,
                y: 0.0,
                width: 800.0,
                height: 600.0,
            },
        }
    }

    #[test]
    fn topology_tools() {
        let reference = plane(4.0, 8);
        let bvh = Bvh::build(&reference);
        let surface = Surface {
            mesh: &reference,
            bvh: &bvh,
        };
        let view = top_view();
        let input = |p: Vec3, modifiers| ToolInput {
            surface,
            view,
            cursor: view.project(p).unwrap(),
            modifiers,
        };
        let none = ModifiersState::empty();
        let click = |tools: &mut Tools, retopo: &mut Retopo, p: Vec3| {
            tools.mouse_move(retopo, &input(p, none));
            tools.mouse_down(retopo, &input(p, none), MouseButton::Left);
            tools.mouse_up(retopo, &input(p, none), MouseButton::Left);
        };

        // Loop: press on an edge, slide along it, release.
        let mut retopo = grid_retopo();
        let mut tools = Tools::default();
        tools.select(ToolKind::Loop);
        let edge_point = |x: f32| Vec3::new(x, 0.0, -1.5);
        tools.mouse_move(&mut retopo, &input(edge_point(-0.9), none));
        assert!(
            !tools.loop_preview(&retopo).is_empty(),
            "a preview while hovering"
        );
        tools.mouse_down(
            &mut retopo,
            &input(edge_point(-0.9), none),
            MouseButton::Left,
        );
        tools.mouse_move(&mut retopo, &input(edge_point(-0.75), none));
        tools.mouse_up(
            &mut retopo,
            &input(edge_point(-0.75), none),
            MouseButton::Left,
        );
        let mesh = retopo.mesh();
        mesh.validate().unwrap();
        assert_eq!((mesh.vertex_count(), mesh.face_count()), (20, 12));
        let slid = mesh
            .vertices()
            .filter(|&v| (mesh.position(v).x + 0.75).abs() < 0.05)
            .count();
        assert_eq!(slid, 4, "the loop went where it was slid to");
        assert_eq!(retopo.doc.undo_name(), Some("Insert edge loop"));

        // Cut: across the middle face, from the middle of one side to the
        // middle of the opposite side.
        let mut retopo = grid_retopo();
        tools.select(ToolKind::Cut);
        click(&mut tools, &mut retopo, Vec3::new(0.0, 0.0, -0.5));
        click(&mut tools, &mut retopo, Vec3::new(0.0, 0.0, 0.5));
        let mesh = retopo.mesh();
        mesh.validate().unwrap();
        assert_eq!((mesh.vertex_count(), mesh.face_count()), (18, 10));
        assert!(tools.status.is_none(), "{:?}", tools.status);

        // Weld: Ctrl+drag a corner onto its neighbor.
        let mut retopo = grid_retopo();
        tools.select(ToolKind::Create);
        let (from, to) = (Vec3::new(-1.5, 0.0, -1.5), Vec3::new(-0.5, 0.0, -1.5));
        let ctrl = ModifiersState::CONTROL;
        tools.mouse_move(&mut retopo, &input(from, ctrl));
        tools.mouse_down(&mut retopo, &input(from, ctrl), MouseButton::Left);
        tools.mouse_move(&mut retopo, &input(to, ctrl));
        tools.mouse_up(&mut retopo, &input(to, ctrl), MouseButton::Left);
        let mesh = retopo.mesh();
        mesh.validate().unwrap();
        assert_eq!(mesh.vertex_count(), 15);
        assert_eq!(retopo.doc.undo_name(), Some("Weld vertices"));
        retopo.undo();
        assert_eq!(
            retopo.mesh().vertex_count(),
            16,
            "drag and weld undo together"
        );
        assert!(
            retopo
                .mesh()
                .vertices()
                .any(|v| retopo.mesh().position(v) == from)
        );

        // Delete the selection; selected ids of removed faces go with them.
        let mut retopo = grid_retopo();
        tools.select(ToolKind::Select);
        tools.set_select_mode(&mut retopo, &input(from, none), SelectMode::Face);
        click(&mut tools, &mut retopo, Vec3::ZERO);
        assert_eq!(retopo.selection.len(), 1);
        tools.key(
            &mut retopo,
            &input(from, none),
            &Key::Named(NamedKey::Delete),
        );
        assert_eq!(retopo.mesh().face_count(), 8);
        assert!(retopo.selection.is_empty());
        retopo.undo();
        assert_eq!(retopo.mesh().face_count(), 9);
    }

    #[test]
    fn building_tools() {
        let reference = plane(4.0, 8);
        let bvh = Bvh::build(&reference);
        let surface = Surface {
            mesh: &reference,
            bvh: &bvh,
        };
        let view = top_view();
        let input = |p: Vec3, modifiers| ToolInput {
            surface,
            view,
            cursor: view.project(p).unwrap(),
            modifiers,
        };
        let none = ModifiersState::empty();
        let drag = |tools: &mut Tools, retopo: &mut Retopo, from: Vec3, to: Vec3| {
            tools.mouse_move(retopo, &input(from, none));
            tools.mouse_down(retopo, &input(from, none), MouseButton::Left);
            tools.mouse_move(retopo, &input(to, none));
            tools.mouse_up(retopo, &input(to, none), MouseButton::Left);
        };
        let edges_at_z = |mesh: &HalfEdgeMesh, z: f32| -> Vec<EdgeId> {
            mesh.edges()
                .filter(|&e| {
                    let h = e.halfedge(0);
                    [mesh.origin(h), mesh.target(h)]
                        .iter()
                        .all(|&v| (mesh.position(v).z - z).abs() < 1e-4)
                })
                .collect()
        };

        // Extrude one border edge outwards, then its new edge again.
        let mut retopo = grid_retopo();
        let mut tools = Tools::default();
        tools.select(ToolKind::Extrude);
        drag(
            &mut tools,
            &mut retopo,
            Vec3::new(-1.0, 0.0, -1.5),
            Vec3::new(-1.0, 0.0, -2.2),
        );
        let mesh = retopo.mesh();
        mesh.validate().unwrap();
        assert_eq!((mesh.vertex_count(), mesh.face_count()), (18, 10));
        let far = mesh
            .vertices()
            .filter(|&v| (mesh.position(v).z + 2.2).abs() < 0.05)
            .count();
        assert_eq!(far, 2, "{tools:?}");
        assert_eq!(retopo.selection.len(), 1, "the new border edge is selected");
        drag(
            &mut tools,
            &mut retopo,
            Vec3::new(-1.0, 0.0, -2.2),
            Vec3::new(-1.0, 0.0, -2.8),
        );
        assert_eq!(
            retopo.mesh().face_count(),
            11,
            "selected edges extrude again"
        );

        // Tweak: the brush drags nearby vertices along, on the surface.
        let mut retopo = grid_retopo();
        tools.select(ToolKind::Tweak);
        drag(
            &mut tools,
            &mut retopo,
            Vec3::new(-0.5, 0.0, -0.5),
            Vec3::new(-0.3, 0.0, -0.5),
        );
        let mesh = retopo.mesh();
        assert!(
            mesh.vertices()
                .any(|v| (mesh.position(v).x + 0.3).abs() < 0.02)
        );
        assert!(mesh.vertices().all(|v| mesh.position(v).y.abs() < 1e-4));
        assert_eq!(retopo.doc.undo_name(), Some("Tweak"));

        // Fill: delete the middle face, select an edge of the hole, fill.
        let mut retopo = grid_retopo();
        tools.select(ToolKind::Select);
        retopo
            .selection
            .set_mode(retopo.doc.mesh(), SelectMode::Face);
        let middle = retopo
            .mesh()
            .faces()
            .find(|&f| {
                !retopo
                    .mesh()
                    .face_vertices(f)
                    .any(|v| retopo.mesh().is_boundary_vertex(v))
            })
            .unwrap();
        retopo
            .selection
            .apply(SelectOp::Replace, Elements::Faces(vec![middle]));
        tools.run(&mut retopo, &input(Vec3::ZERO, none), Command::Delete);
        assert_eq!(retopo.mesh().face_count(), 8);
        retopo
            .selection
            .set_mode(retopo.doc.mesh(), SelectMode::Edge);
        let hole_edge = edges_at_z(retopo.mesh(), -0.5)
            .into_iter()
            .find(|&e| retopo.mesh().is_boundary_edge(e))
            .unwrap();
        retopo
            .selection
            .apply(SelectOp::Replace, Elements::Edges(vec![hole_edge]));
        tools.run(&mut retopo, &input(Vec3::ZERO, none), Command::Fill);
        assert_eq!(retopo.mesh().face_count(), 9, "{:?}", tools.status);
        retopo.mesh().validate().unwrap();

        // Bridge: remove the middle row, then bridge the two borders.
        let mut retopo = grid_retopo();
        retopo
            .selection
            .set_mode(retopo.doc.mesh(), SelectMode::Face);
        let row: Vec<FaceId> = retopo
            .mesh()
            .faces()
            .filter(|&f| {
                let mesh = retopo.mesh();
                let z: f32 = mesh
                    .face_vertices(f)
                    .map(|v| mesh.position(v).z)
                    .sum::<f32>()
                    / 4.0;
                z.abs() < 1e-4
            })
            .collect();
        assert_eq!(row.len(), 3);
        retopo
            .selection
            .apply(SelectOp::Replace, Elements::Faces(row));
        tools.run(&mut retopo, &input(Vec3::ZERO, none), Command::Delete);
        assert_eq!(retopo.mesh().face_count(), 6);
        retopo
            .selection
            .set_mode(retopo.doc.mesh(), SelectMode::Edge);
        let mut borders = edges_at_z(retopo.mesh(), -0.5);
        borders.extend(edges_at_z(retopo.mesh(), 0.5));
        retopo
            .selection
            .apply(SelectOp::Replace, Elements::Edges(borders));
        tools.run(&mut retopo, &input(Vec3::ZERO, none), Command::Bridge);
        assert_eq!(retopo.mesh().face_count(), 9, "{:?}", tools.status);
        retopo.mesh().validate().unwrap();

        // Hard edges: around a selected corner face, and its mirror image.
        let mut retopo = grid_retopo();
        retopo.symmetry_axis = Some(Axis::X);
        retopo.scene_radius = 4.0;
        retopo
            .selection
            .set_mode(retopo.doc.mesh(), SelectMode::Face);
        let corner = retopo
            .mesh()
            .faces()
            .find(|&f| {
                let mesh = retopo.mesh();
                mesh.face_vertices(f)
                    .any(|v| mesh.position(v) == Vec3::new(-1.5, 0.0, -1.5))
            })
            .unwrap();
        retopo
            .selection
            .apply(SelectOp::Replace, Elements::Faces(vec![corner]));
        tools.run(&mut retopo, &input(Vec3::ZERO, none), Command::MarkHard);
        let mesh = retopo.mesh();
        assert_eq!(mesh.hard_edges().count(), 8, "{:?}", tools.status);
        assert!(
            mesh.hard_edges()
                .all(|e| mesh.position(mesh.origin(e.halfedge(0))).x.abs() > 0.4)
        );
        assert_eq!(retopo.doc.undo_name(), Some("Mark hard"));
        let shift = ModifiersState::SHIFT;
        let key = Key::Character("k".into());
        assert!(tools.key(&mut retopo, &input(Vec3::ZERO, shift), &key));
        assert!(!retopo.mesh().has_hard_edges());
        retopo.undo();
        assert_eq!(retopo.mesh().hard_edges().count(), 8);
    }

    #[test]
    fn box_select_then_loop() {
        use polyflow_core::camera::OrbitCamera;
        use polyflow_render::Viewport;

        let reference = plane(4.0, 8);
        let bvh = Bvh::build(&reference);
        let surface = Surface {
            mesh: &reference,
            bvh: &bvh,
        };
        let mut retopo = Retopo::default();
        retopo
            .doc
            .edit("Grid", |m: &mut HalfEdgeMesh| {
                let ids: Vec<VertexId> = (0..16)
                    .map(|k| {
                        let (i, j) = ((k % 4) as f32, (k / 4) as f32);
                        m.add_vertex(Vec3::new(i - 1.5, 0.0, j - 1.5))
                    })
                    .collect();
                for j in 0..3 {
                    for i in 0..3 {
                        let k = j * 4 + i;
                        let quad = [ids[k], ids[k + 1], ids[k + 5], ids[k + 4]];
                        edit::add_face_facing(m, &quad, Vec3::Y).unwrap();
                    }
                }
                Ok::<_, ()>(())
            })
            .unwrap();
        let view = ViewInfo {
            camera: OrbitCamera {
                pitch: 1.2,
                distance: 8.0,
                ..Default::default()
            },
            viewport: Viewport {
                x: 0.0,
                y: 0.0,
                width: 800.0,
                height: 600.0,
            },
        };
        let input = |cursor, modifiers| ToolInput {
            surface,
            view,
            cursor,
            modifiers,
        };
        let screen: Vec<Vec2> = retopo
            .mesh()
            .vertices()
            .map(|v| view.project(retopo.mesh().position(v)).unwrap())
            .collect();
        let lo = screen.iter().fold(Vec2::MAX, |a, &b| a.min(b)) - 5.0;
        let hi = screen.iter().fold(Vec2::MIN, |a, &b| a.max(b)) + 5.0;
        let none = ModifiersState::empty();

        let mut tools = Tools::default();
        tools.select(ToolKind::Select);
        let drag = |tools: &mut Tools, retopo: &mut Retopo, modifiers| {
            assert!(tools.mouse_down(retopo, &input(lo, modifiers), MouseButton::Left));
            tools.mouse_move(retopo, &input(hi, modifiers));
            assert!(tools.is_busy());
            tools.mouse_up(retopo, &input(hi, modifiers), MouseButton::Left);
        };
        drag(&mut tools, &mut retopo, none);
        assert_eq!(retopo.selection.len(), 16);

        tools.set_select_mode(&mut retopo, &input(hi, none), SelectMode::Face);
        assert_eq!(retopo.selection.len(), 9, "converted to faces");
        drag(&mut tools, &mut retopo, ModifiersState::CONTROL);
        assert!(retopo.selection.is_empty(), "Ctrl removes");

        // A click on a face, then a double-click near its side: the row.
        let center = |f| {
            let mesh = retopo.mesh();
            let sum: Vec3 = mesh.face_vertices(f).map(|v| mesh.position(v)).sum();
            view.project(sum / 4.0).unwrap()
        };
        let middle = retopo
            .mesh()
            .faces()
            .find(|&f| {
                !retopo
                    .mesh()
                    .face_vertices(f)
                    .any(|v| retopo.mesh().is_boundary_vertex(v))
            })
            .unwrap();
        let at = center(middle);
        for _ in 0..2 {
            tools.mouse_down(&mut retopo, &input(at, none), MouseButton::Left);
            tools.mouse_up(&mut retopo, &input(at, none), MouseButton::Left);
        }
        assert_eq!(retopo.selection.len(), 3, "a face loop across the grid");
        assert!(retopo.selection.faces().contains(&middle));
    }
}
