//! Polyflow desktop application.

// Release builds are GUI applications on Windows: no console window opens
// next to the app. Debug builds keep the console for logs.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod bake;
mod bench;
mod camera_control;
mod jobs;
mod modal;
mod options;
mod project;
mod retopo;
mod ui;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use polyflow_core::camera::OrbitCamera;
use polyflow_core::glam::{Vec2, Vec3};
use polyflow_core::mesh::TriMesh;
use polyflow_render::{FrameInput, MeshView, Overlay, OverlayStyle, Renderer, UiFrame, Viewport};
use polyflow_spatial::Bvh;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::Key;
use winit::window::{Window, WindowId};

use crate::bench::Bench;
use crate::camera_control::CameraController;
use crate::jobs::{Job, Prepared};
use crate::options::Options;
use crate::project::Autosave;
use crate::retopo::Retopo;
use crate::retopo::surface::Surface;
use crate::retopo::symmetry::Axis;
use crate::retopo::tools::{self, ToolInput, ToolKind, Tools};
use crate::retopo::view::ViewInfo;
use crate::ui::{Actions, Hover, MeshInfo, RetopoUi, SelectAction, Stats, UiState};

/// Pixels of a high-resolution wheel/touchpad delta that count as one wheel line.
const PIXELS_PER_SCROLL_LINE: f32 = 50.0;
/// Linear RGB.
const BACKGROUND: Vec3 = Vec3::new(0.018, 0.020, 0.024);

/// Builds a [`ToolInput`] from `AppState` fields, borrowing only the
/// reference mesh so tools and the retopo state can be borrowed mutably.
macro_rules! tool_input {
    ($state:expr) => {
        ToolInput {
            surface: Surface {
                mesh: &$state.mesh,
                bvh: &$state.bvh,
            },
            view: ViewInfo {
                camera: $state.camera,
                viewport: $state.viewport,
            },
            cursor: $state.controller.cursor().unwrap_or_default(),
            modifiers: $state.controller.modifiers,
        }
    };
}

/// Events sent to the event loop from other threads.
#[derive(Debug)]
enum UserEvent {
    MeshReady(Box<Result<Prepared, String>>),
    DialogAnswer(modal::Answer),
    BakeDone(Box<bake::BakeOutcome>),
}

fn main() -> anyhow::Result<()> {
    attach_parent_console();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let options = match options::parse(std::env::args_os().skip(1)) {
        options::ParseResult::Run(options) => options,
        options::ParseResult::Help => {
            println!("{}", options::USAGE);
            return Ok(());
        }
        options::ParseResult::Error(e) => {
            eprintln!("error: {e}\n\n{}", options::USAGE);
            std::process::exit(2);
        }
    };
    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let mut app = App {
        state: None,
        error: None,
        proxy: event_loop.create_proxy(),
        options: Some(options),
    };
    event_loop.run_app(&mut app)?;
    app.error.map_or(Ok(()), Err)
}

struct App {
    state: Option<AppState>,
    error: Option<anyhow::Error>,
    proxy: EventLoopProxy<UserEvent>,
    /// Command-line options, applied once the window exists.
    options: Option<Options>,
}

impl App {
    fn handle(&mut self, event_loop: &ActiveEventLoop, result: anyhow::Result<()>) {
        if let Err(e) = result {
            self.error = Some(e);
            event_loop.exit();
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let options = self.options.take().unwrap_or_default();
        match AppState::new(event_loop, self.proxy.clone(), &options) {
            Ok(mut state) => {
                let mesh = options.mesh;
                let start = move |state: &mut AppState| match mesh {
                    Some(path) if is_project(&path) => state.open_project(&path, false),
                    Some(path) => state.start_job(Job::Open(path)),
                    None => state.start_job(Job::Torus {
                        rings: state.ui.torus_rings,
                    }),
                };
                if state.autosave.enabled {
                    state.offer_recovery(start);
                } else {
                    start(&mut state);
                }
                self.state = Some(state);
            }
            Err(e) => self.handle(event_loop, Err(e)),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        let result = state.window_event(event_loop, event);
        self.handle(event_loop, result);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        let Some(state) = &mut self.state else { return };
        let result = match event {
            UserEvent::MeshReady(result) => state.mesh_ready(*result),
            UserEvent::DialogAnswer(answer) => {
                state.dialog_answered(answer);
                Ok(())
            }
            UserEvent::BakeDone(outcome) => {
                state.bake_done(*outcome);
                Ok(())
            }
        };
        self.handle(event_loop, result);
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Tear down the renderer and window while the display connection is
        // still open: after `run_app` returns it is gone (on Wayland,
        // destroying the Vulkan surface then crashes the driver).
        if let Some(state) = &mut self.state {
            // A clean exit: the user saved or chose to discard.
            state.remove_autosave();
            if let Some(job) = &state.bake_job {
                job.control.cancel();
            }
        }
        self.state = None;
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = &mut self.state else { return };
        if state.exit_requested {
            event_loop.exit();
            return;
        }
        let autosave_at = state.autosave_if_due();
        let repaint_at = match state.next_repaint {
            Some(at) if at <= Instant::now() => {
                state.next_repaint = None;
                state.window.request_redraw();
                None
            }
            other => other,
        };
        match repaint_at.into_iter().chain(autosave_at).min() {
            Some(at) => event_loop.set_control_flow(ControlFlow::WaitUntil(at)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }
}

struct AppState {
    // Declared before `window`: the renderer must be dropped while the window still exists.
    renderer: Renderer,
    window: Arc<Window>,
    proxy: EventLoopProxy<UserEvent>,
    egui_state: egui_winit::State,
    camera: OrbitCamera,
    controller: CameraController,
    ui: UiState,
    /// CPU copy of the reference mesh, for picking, snapping and baking
    /// (shared with bake threads).
    mesh: Arc<TriMesh>,
    bvh: Arc<Bvh>,
    mesh_info: MeshInfo,
    /// Surface under the cursor, updated as the cursor or camera moves.
    hover: Option<Hover>,
    /// Description of the mesh being prepared on a background thread.
    loading: Option<String>,
    /// Region of the window showing the 3D scene, in physical pixels.
    viewport: Viewport,
    next_repaint: Option<Instant>,
    last_frame: Option<Instant>,
    cpu_ms: f32,
    interval_ms: f32,
    gpu_ms: Option<f32>,
    /// Requested `--bench` duration; the benchmark starts once a mesh is shown.
    bench_duration: Option<Duration>,
    bench: Option<Bench>,
    /// `--zoom` factor, applied when the first mesh is framed.
    initial_zoom: Option<f32>,
    retopo: Retopo,
    tools: Tools,
    /// Hover highlights and tool previews, rebuilt every frame.
    tool_overlay: Overlay,
    /// The open project file, if it was saved or opened.
    project_path: Option<PathBuf>,
    /// The reference mesh file (`None` for the generated test mesh).
    reference_path: Option<PathBuf>,
    /// Retopo mesh version and reference at the last save, to detect
    /// unsaved changes.
    saved_version: u64,
    saved_reference: Option<PathBuf>,
    /// Camera to restore once the project's reference mesh has loaded.
    pending_camera: Option<polyflow_io::project::CameraState>,
    autosave: Autosave,
    /// The dialog being shown, if any.
    dialogs: modal::Dialogs,
    /// Bake panel settings, saved with the project.
    bake_state: polyflow_io::project::BakeState,
    bake_job: Option<bake::BakeJob>,
    /// The last finished bake, for the panel's summary and the preview.
    last_bake: Option<Box<bake::BakeReport>>,
    /// How the last bake is previewed, if it is.
    preview_mode: Option<polyflow_render::PreviewMode>,
    /// Keep drawing the sculpt while previewing.
    preview_show_sculpt: bool,
    /// Set once the user agreed to close; the event loop exits next.
    exit_requested: bool,
    /// Title currently shown, to avoid resetting it every frame.
    title: String,
}

impl AppState {
    fn new(
        event_loop: &ActiveEventLoop,
        proxy: EventLoopProxy<UserEvent>,
        options: &Options,
    ) -> anyhow::Result<Self> {
        let attributes = Window::default_attributes()
            .with_title("Polyflow")
            .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 900.0));
        let window = Arc::new(event_loop.create_window(attributes)?);
        let size = window.inner_size();
        let mut ui = UiState {
            vsync: options.vsync,
            continuous: options.continuous,
            ..Default::default()
        };
        if let Some(rings) = options.torus_rings {
            ui.torus_rings = rings;
        }
        let renderer = Renderer::new(window.as_ref(), size.width, size.height, ui.vsync)
            .context("failed to initialize Vulkan")?;

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(renderer.max_texture_side()),
        );

        let state = Self {
            renderer,
            window,
            proxy,
            egui_state,
            camera: OrbitCamera::default(),
            controller: CameraController::default(),
            mesh: Arc::new(TriMesh::default()),
            bvh: Arc::new(Bvh::default()),
            mesh_info: MeshInfo::empty(),
            hover: None,
            loading: None,
            viewport: Viewport {
                width: size.width as f32,
                height: size.height as f32,
                ..Default::default()
            },
            next_repaint: None,
            last_frame: None,
            cpu_ms: 0.0,
            interval_ms: 0.0,
            gpu_ms: None,
            bench_duration: options.bench,
            bench: None,
            initial_zoom: options.zoom,
            retopo: Retopo::default(),
            tools: Tools::default(),
            tool_overlay: Overlay::default(),
            project_path: None,
            reference_path: None,
            saved_version: 0,
            saved_reference: None,
            pending_camera: None,
            autosave: Autosave::new(options.bench.is_none()),
            bake_state: polyflow_io::project::BakeState::default(),
            bake_job: None,
            last_bake: None,
            preview_mode: None,
            preview_show_sculpt: false,
            dialogs: modal::Dialogs::default(),
            exit_requested: false,
            title: String::new(),
            ui,
        };
        Ok(state)
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        event: WindowEvent,
    ) -> anyhow::Result<()> {
        // A dialog is modal: the window keeps drawing but takes no input.
        if self.dialogs.is_open() && modal::is_blocked_input(&event) {
            return Ok(());
        }
        let response = self.egui_state.on_window_event(&self.window, &event);
        if response.repaint {
            self.window.request_redraw();
        }
        let viewport_height = self.viewport.height;

        match event {
            WindowEvent::CloseRequested => self.request_close(),
            WindowEvent::Resized(size) => {
                self.renderer.resize(size.width, size.height);
                self.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let actions = self.redraw()?;
                self.apply_post_frame_actions(&actions);
                self.update_title();
            }
            WindowEvent::DroppedFile(path) if is_project(&path) => {
                self.confirm_discard(
                    "Opening a project replaces the current work.",
                    move |state| state.open_project(&path, false),
                );
            }
            WindowEvent::DroppedFile(path) => self.start_job(Job::Open(path)),
            WindowEvent::ModifiersChanged(modifiers) => {
                self.controller.modifiers = modifiers.state();
                // Some tool previews depend on the modifiers (Loop + Shift).
                if !self.tools.is_busy() && !self.pointer_over_ui() {
                    let input = tool_input!(self);
                    self.tools.update_hover(&self.retopo, &input);
                }
                self.window.request_redraw();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                let over_ui = self.pointer_over_ui();
                let started =
                    !self.tools.is_busy() && self.controller.mouse_button(button, pressed, over_ui);
                if !pressed {
                    // Releases always reach both, so no gesture gets stuck.
                    self.controller.mouse_button(button, false, over_ui);
                    let input = tool_input!(self);
                    self.tools.mouse_up(&mut self.retopo, &input, button);
                } else if started {
                    if self.ui.auto_depth {
                        self.update_hover();
                        if let Some(hover) = self.hover {
                            self.camera.set_pivot_depth(hover.point);
                        }
                    }
                } else if !over_ui
                    && !self.controller.is_dragging()
                    && !self.controller.modifiers.alt_key()
                {
                    let input = tool_input!(self);
                    self.tools.mouse_down(&mut self.retopo, &input, button);
                }
                self.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = Vec2::new(position.x as f32, position.y as f32);
                if self
                    .controller
                    .cursor_moved(position, &mut self.camera, viewport_height)
                {
                    self.window.request_redraw();
                }
                if !self.controller.is_dragging() {
                    self.update_hover();
                    if self.pointer_over_ui() && !self.tools.is_busy() {
                        self.tools.clear_hover();
                    } else {
                        let input = tool_input!(self);
                        self.tools.mouse_move(&mut self.retopo, &input);
                    }
                    self.window.request_redraw();
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.hover = None;
                self.tools.clear_hover();
                self.window.request_redraw();
            }
            WindowEvent::MouseWheel { delta, .. } if !self.pointer_over_ui() => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / PIXELS_PER_SCROLL_LINE,
                };
                let factor = CameraController::wheel_factor(lines);
                self.update_hover();
                match self.hover {
                    Some(hover) if self.ui.zoom_to_cursor => {
                        self.camera.zoom_about(hover.point, factor)
                    }
                    _ => self.camera.zoom(factor),
                }
                self.update_hover();
                let input = tool_input!(self);
                self.tools.update_hover(&self.retopo, &input);
                self.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !response.consumed && !event.repeat =>
            {
                self.key_pressed(event_loop, &event.logical_key);
                self.window.request_redraw();
            }
            _ => {}
        }
        Ok(())
    }

    fn key_pressed(&mut self, _event_loop: &ActiveEventLoop, key: &Key) {
        let mods = self.controller.modifiers;
        let ctrl = mods.control_key();
        let character = match key {
            Key::Character(c) => Some(c.to_ascii_lowercase()),
            _ => None,
        };
        match character.as_deref() {
            Some("z") if ctrl && mods.shift_key() => self.redo(),
            Some("z") if ctrl => self.undo(),
            Some("y") if ctrl => self.redo(),
            Some("n") if ctrl => self.new_project(),
            Some("o") if ctrl => self.open_project_dialog(),
            Some("l") if ctrl => self.open_file_dialog(),
            Some("i") if ctrl => self.import_retopo(),
            Some("e") if ctrl => self.export_retopo(),
            Some("b") if ctrl => self.start_bake(),
            Some("s") if ctrl => {
                self.save_project(mods.shift_key(), |_| {});
            }
            Some("f") if !ctrl => self.frame_mesh(),
            Some("s") if !ctrl => self.tools.select(ToolKind::Select),
            Some("q") if !ctrl => self.tools.select(ToolKind::Create),
            Some("d") if !ctrl => self.tools.select(ToolKind::Draw),
            Some("r") if !ctrl => self.tools.select(ToolKind::Relax),
            Some("i") if !ctrl => self.tools.select(ToolKind::Loop),
            Some("c") if !ctrl => self.tools.select(ToolKind::Cut),
            Some("e") if !ctrl => self.tools.select(ToolKind::Extrude),
            Some("w") if !ctrl => self.tools.select(ToolKind::Tweak),
            Some("x") if !ctrl => {
                self.retopo.symmetry_axis = match self.retopo.symmetry_axis {
                    Some(Axis::X) => None,
                    _ => Some(Axis::X),
                };
            }
            _ => {
                let input = tool_input!(self);
                self.tools.key(&mut self.retopo, &input, key);
            }
        }
    }

    fn undo(&mut self) {
        self.retopo.undo();
        self.tools.status = None;
        let input = tool_input!(self);
        self.tools.update_hover(&self.retopo, &input);
    }

    fn redo(&mut self) {
        self.retopo.redo();
        self.tools.status = None;
        let input = tool_input!(self);
        self.tools.update_hover(&self.retopo, &input);
    }

    fn relax_all(&mut self) {
        let vertices: Vec<_> = self.retopo.mesh().vertices().collect();
        let weights = vec![self.tools.settings.brush_strength; vertices.len()];
        let surface = Surface {
            mesh: &self.mesh,
            bvh: &self.bvh,
        };
        let borders = self.tools.settings.relax_borders;
        tools::relax(
            &mut self.retopo,
            &surface,
            &vertices,
            &weights,
            borders,
            None,
        );
    }

    fn clear_retopo(&mut self) {
        let _ = self.retopo.edit("Clear retopology", |m| {
            let faces: Vec<_> = m.faces().collect();
            for f in faces {
                m.remove_face(f, false)?;
            }
            let vertices: Vec<_> = m.vertices().collect();
            for v in vertices {
                m.remove_vertex(v)?;
            }
            Ok::<_, polyflow_core::halfedge::TopologyError>(())
        });
    }

    /// Prepares a mesh on a background thread; the result arrives as a
    /// [`UserEvent::MeshReady`].
    fn start_job(&mut self, job: Job) {
        self.window.request_redraw();
        if let Some(current) = &self.loading {
            self.ui.error = Some(format!("Still preparing {current}"));
            return;
        }
        log::info!("Preparing {}", job.name());
        self.loading = Some(job.name());
        let proxy = self.proxy.clone();
        let spawned = std::thread::Builder::new()
            .name("mesh preparation".into())
            .spawn(move || {
                let result = Box::new(jobs::run(&job));
                // Fails only if the event loop is gone, i.e. the app is closing.
                let _ = proxy.send_event(UserEvent::MeshReady(result));
            });
        if let Err(e) = spawned {
            self.loading = None;
            self.ui.error = Some(format!("Could not start a background thread: {e}"));
        }
    }

    fn mesh_ready(&mut self, result: Result<Prepared, String>) -> anyhow::Result<()> {
        self.loading = None;
        self.window.request_redraw();
        let prepared = match result {
            Ok(prepared) => prepared,
            Err(message) => {
                log::error!("{message}");
                self.ui.error = Some(message);
                return Ok(());
            }
        };

        let start = Instant::now();
        self.renderer.set_mesh(&prepared.gpu_data)?;
        let mut info = prepared.info;
        info.upload = start.elapsed();
        log::info!(
            "Showing {}: {} vertices, {} triangles, {} meshlets (meshlets {:?}, BVH {:?} / {} MB, upload {:?})",
            info.name,
            info.vertices,
            info.triangles,
            info.meshlets,
            info.meshlet_build,
            info.bvh_build,
            info.bvh_bytes >> 20,
            info.upload,
        );
        self.mesh_info = info;
        self.reference_path = prepared.source_path;
        self.mesh = Arc::new(prepared.mesh);
        self.bvh = Arc::new(prepared.bvh);
        self.retopo.scene_radius = self.mesh.bounds().radius().max(1e-6);
        self.hover = None;
        match self.pending_camera.take() {
            Some(camera) => self.apply_camera(&camera),
            None => self.frame_mesh(),
        }
        if let Some(factor) = self.initial_zoom.take() {
            self.camera.zoom(1.0 / factor);
        }
        Ok(())
    }

    fn open_file_dialog(&mut self) {
        self.ask(
            |parent| {
                rfd::FileDialog::new()
                    .set_title("Open reference mesh")
                    .add_filter("Meshes", &["obj"])
                    .add_filter("All files", &["*"])
                    .set_parent(parent)
                    .pick_file()
            },
            |state, picked| {
                if let Some(path) = picked {
                    state.start_job(Job::Open(path));
                }
            },
        );
    }

    /// Raycasts the reference mesh under the cursor.
    fn update_hover(&mut self) {
        self.hover = None;
        let Some(cursor) = self.controller.cursor() else {
            return;
        };
        if self.pointer_over_ui() {
            return;
        }
        let v = self.viewport;
        let ndc = Vec2::new(
            (cursor.x - v.x) / v.width * 2.0 - 1.0,
            (cursor.y - v.y) / v.height * 2.0 - 1.0,
        );
        let (origin, dir) = self.camera.ray(ndc, self.viewport_aspect());
        let start = Instant::now();
        let hit = self.bvh.raycast(&self.mesh, origin, dir, f32::INFINITY);
        let query = start.elapsed();
        self.hover = hit.map(|hit| Hover {
            point: origin + dir * hit.t,
            triangle: hit.triangle,
            query,
        });
    }

    fn frame_mesh(&mut self) {
        let aspect = self.viewport_aspect();
        self.camera.frame(&self.mesh.bounds(), aspect);
    }

    /// Runs the UI and renders a frame. Actions that block (dialogs) or end
    /// the app are returned to run after the frame is presented.
    fn redraw(&mut self) -> anyhow::Result<Actions> {
        let start = Instant::now();
        let mut frame_interval_ms = None;
        if let Some(last) = self.last_frame.replace(start) {
            let ms = (start - last).as_secs_f32() * 1000.0;
            smooth(&mut self.interval_ms, ms);
            frame_interval_ms = Some(ms);
            if self.bench.is_some() {
                // Turntable at a fixed angular speed, independent of frame rate.
                self.camera.orbit(ms * 0.0005, 0.0);
            }
        }
        let gpu_ms = self
            .renderer
            .gpu_frame_time()
            .map(|d| d.as_secs_f32() * 1000.0);
        if let Some(ms) = gpu_ms {
            smooth(self.gpu_ms.get_or_insert(ms), ms);
        }

        let cursor_in_viewport = !self.pointer_over_ui();
        let stats = Stats {
            gpu: self.renderer.device_name(),
            present_mode: self.renderer.present_mode_name(),
            mesh: &self.mesh_info,
            loading: self.loading.as_deref(),
            culling: self.renderer.cull_stats(),
            hover: self.hover.as_ref(),
            cpu_ms: self.cpu_ms,
            interval_ms: self.interval_ms,
            gpu_ms: self.gpu_ms,
            brush: {
                let input = tool_input!(self);
                self.tools.brush_circle(&input, cursor_in_viewport)
            },
            region: self.tools.region_outline(),
            cursor_in_viewport,
        };
        let retopo_stats = self.retopo.stats();
        let bake_blocker = self.bake_blocker(retopo_stats.uv_faces);
        let cage_imported = self
            .retopo
            .imported_cage()
            .map(|_| !self.retopo.imported_cage_is_stale());
        let auto_cage_push = self.retopo.scene_radius * retopo::DEFAULT_CAGE_PUSH;
        let scene_radius = self.retopo.scene_radius;
        let mut select_mode = self.retopo.selection.mode();
        let mut retopo_ui = RetopoUi {
            tool: &mut self.tools.active,
            select_mode: &mut select_mode,
            lasso: &mut self.tools.lasso,
            selected: self.retopo.selection.len(),
            settings: &mut self.tools.settings,
            symmetry: &mut self.retopo.symmetry_axis,
            x_ray: &mut self.retopo.x_ray,
            surface_offset: &mut self.retopo.surface_offset,
            uv_checker: &mut self.retopo.uv_checker,
            show_cage: &mut self.retopo.show_cage,
            cage_push: &mut self.retopo.cage_push,
            auto_cage_push,
            scene_radius,
            cage_imported,
            status: &mut self.tools.status,
            undo_name: self.retopo.doc.undo_name(),
            redo_name: self.retopo.doc.redo_name(),
            vertices: self.retopo.doc.mesh().vertex_count(),
            faces: self.retopo.doc.mesh().face_count(),
            face_kinds: retopo_stats.face_kinds,
            uv_faces: retopo_stats.uv_faces,
        };
        let bake_output = bake::output_prefix(
            &self.bake_state,
            self.project_path.as_deref(),
            self.reference_path.as_deref(),
        );
        let last_bake = self.last_bake.as_ref().map(|r| {
            let maps = r.files.len().saturating_sub(1);
            let covered = r.baked.hits + r.baked.misses;
            let missed = 100.0 * r.baked.misses as f64 / covered.max(1) as f64;
            format!(
                "Baked {maps} map{} in {:.1} s on {}; {missed:.2}% of texels missed.",
                if maps == 1 { "" } else { "s" },
                r.time.as_secs_f64(),
                r.device
            )
        });
        let mut bake_ui = ui::BakeUi {
            state: &mut self.bake_state,
            scene_radius: self.retopo.scene_radius,
            output: bake_output.display().to_string(),
            progress: self
                .bake_job
                .as_ref()
                .map(|j| (j.progress(), j.started.elapsed())),
            blocker: bake_blocker,
            last: last_bake,
            has_preview: self.last_bake.is_some(),
            preview: &mut self.preview_mode,
            show_sculpt: &mut self.preview_show_sculpt,
        };
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let mut actions = Actions::default();
        let mut viewport_rect = egui::Rect::NOTHING;
        let ui_state = &mut self.ui;
        let output = self.egui_state.egui_ctx().run_ui(raw_input, |ui| {
            (actions, viewport_rect) = ui_state.show(ui, &stats, &mut retopo_ui, &mut bake_ui)
        });
        self.egui_state
            .handle_platform_output(&self.window, output.platform_output);
        let primitives = self
            .egui_state
            .egui_ctx()
            .tessellate(output.shapes, output.pixels_per_point);

        let ppp = output.pixels_per_point;
        self.viewport = Viewport {
            x: viewport_rect.min.x * ppp,
            y: viewport_rect.min.y * ppp,
            width: viewport_rect.width().max(0.0) * ppp,
            height: viewport_rect.height().max(0.0) * ppp,
        };
        if actions.regenerate_mesh {
            self.start_job(Job::Torus {
                rings: self.ui.torus_rings,
            });
        }
        if actions.frame_mesh {
            self.frame_mesh();
        }
        if actions.undo {
            self.undo();
        }
        if actions.redo {
            self.redo();
        }
        if actions.relax_all {
            self.relax_all();
        }
        if actions.clear_retopo {
            self.clear_retopo();
        }
        if select_mode != self.retopo.selection.mode() {
            let input = tool_input!(self);
            self.tools
                .set_select_mode(&mut self.retopo, &input, select_mode);
        }
        let mesh = self.retopo.doc.mesh();
        match actions.select {
            Some(SelectAction::All) => self.retopo.selection.select_all(mesh),
            Some(SelectAction::None) => self.retopo.selection.clear(),
            Some(SelectAction::Invert) => self.retopo.selection.invert(mesh),
            Some(SelectAction::Loops) => self.tools.grow_loops(&mut self.retopo, false),
            Some(SelectAction::Rings) => self.tools.grow_loops(&mut self.retopo, true),
            None => {}
        }
        if let Some(command) = actions.command {
            let input = tool_input!(self);
            self.tools.run(&mut self.retopo, &input, command);
        }
        self.tools.overlay(&self.retopo, &mut self.tool_overlay);
        let x_ray = self.retopo.x_ray;
        let surface_offset = self.retopo.surface_offset;
        let view_proj = self.camera.view_projection(self.viewport_aspect());
        self.retopo.update_overlays();
        let preview = self.preview_mode.filter(|_| self.last_bake.is_some());
        // The preview shows the low-poly as baked; the retopology would hide it.
        let mut overlays = vec![&self.tool_overlay];
        if preview.is_none() {
            overlays.push(self.retopo.mesh_overlay());
            // After the mesh, so selected edges draw over its edges.
            overlays.push(self.retopo.selection_overlay());
            overlays.extend(self.retopo.cage_overlay());
        }
        self.renderer.set_vsync(self.ui.vsync);

        let [r, g, b] = self.ui.mesh_color;
        self.renderer.render(FrameInput {
            viewport: self.viewport,
            view: MeshView {
                view_proj,
                eye: self.camera.eye(),
                color: Vec3::new(r, g, b),
                backface_culling: self.ui.backface_culling,
                show_meshlets: self.ui.show_meshlets,
            },
            clear_color: BACKGROUND,
            overlays: &overlays,
            show_reference: preview.is_none() || self.preview_show_sculpt,
            preview,
            overlay_style: OverlayStyle {
                line_width: 1.5 * ppp,
                x_ray,
                surface_offset,
            },
            ui: Some(UiFrame {
                primitives: &primitives,
                textures: output.textures_delta,
                pixels_per_point: output.pixels_per_point,
            }),
        })?;

        let repaint_delay = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map_or(Duration::MAX, |v| v.repaint_delay);
        if self.ui.continuous || repaint_delay.is_zero() || actions.frame_mesh {
            self.window.request_redraw();
        } else {
            // Keep the progress bar moving while baking.
            let delay = if self.bake_job.is_some() {
                repaint_delay.min(Duration::from_millis(100))
            } else {
                repaint_delay
            };
            self.next_repaint = Instant::now().checked_add(delay);
            // Frame intervals are only meaningful while redrawing back to back.
            self.last_frame = None;
        }
        let cpu_ms = start.elapsed().as_secs_f32() * 1000.0;
        smooth(&mut self.cpu_ms, cpu_ms);
        self.record_bench(cpu_ms, gpu_ms, frame_interval_ms, &mut actions);
        Ok(actions)
    }

    fn record_bench(
        &mut self,
        cpu_ms: f32,
        gpu_ms: Option<f32>,
        frame_ms: Option<f32>,
        actions: &mut Actions,
    ) {
        // Start timing once the first mesh is on screen.
        if self.bench.is_none() && self.loading.is_none() && self.mesh_info.triangles > 0 {
            self.bench = self.bench_duration.take().map(Bench::new);
        }
        let Some(bench) = &mut self.bench else { return };
        if bench.record(cpu_ms, gpu_ms, frame_ms) {
            let culling = self.renderer.cull_stats().unwrap_or_default();
            println!(
                "{} vertices, {} triangles, {} meshlets ({} visible in the last frame), {}x{} viewport\n{}",
                self.mesh_info.vertices,
                self.mesh_info.triangles,
                self.mesh_info.meshlets,
                culling.visible_meshlets,
                self.viewport.width,
                self.viewport.height,
                bench.report()
            );
            actions.quit = true;
        }
    }

    fn apply_post_frame_actions(&mut self, actions: &Actions) {
        if actions.quit {
            self.request_close();
        } else if actions.open_file {
            self.open_file_dialog();
        } else if actions.new_project {
            self.new_project();
        } else if actions.open_project {
            self.open_project_dialog();
        } else if actions.save_project || actions.save_project_as {
            self.save_project(actions.save_project_as, |_| {});
        } else if actions.import_retopo {
            self.import_retopo();
        } else if actions.export_retopo {
            self.export_retopo();
        } else if actions.bake {
            self.start_bake();
        } else if actions.cancel_bake {
            if let Some(job) = &self.bake_job {
                job.control.cancel();
            }
        } else if actions.choose_bake_output {
            self.choose_bake_output();
        } else if actions.import_cage {
            self.import_cage();
        } else if actions.export_cage {
            self.export_cage();
        } else if actions.clear_cage {
            self.retopo.set_imported_cage(None);
            self.saved_version = u64::MAX;
        }
    }

    fn viewport_aspect(&self) -> f32 {
        self.viewport.width.max(1.0) / self.viewport.height.max(1.0)
    }

    /// Whether the cursor is over the UI rather than the 3D viewport.
    ///
    /// Decided from the latest cursor position instead of egui's own pointer
    /// state, which only updates when a frame runs and can lag behind a quick
    /// move from the panel into the viewport followed by a click.
    fn pointer_over_ui(&self) -> bool {
        let Some(pos) = self.controller.cursor() else {
            return true;
        };
        let v = self.viewport;
        let in_viewport =
            pos.x >= v.x && pos.y >= v.y && pos.x < v.x + v.width && pos.y < v.y + v.height;
        let ctx = self.egui_state.egui_ctx();
        if !in_viewport || ctx.egui_is_using_pointer() {
            return true;
        }
        // Popups and windows (e.g. the color picker) can overlap the viewport.
        let ppp = ctx.pixels_per_point();
        ctx.layer_id_at(egui::pos2(pos.x / ppp, pos.y / ppp))
            .is_some_and(|layer| layer.order != egui::Order::Background)
    }
}

/// Without a console of its own (release builds on Windows), reuse the
/// terminal the app was started from, if any, so logs and `--bench` output
/// still show up there.
fn attach_parent_console() {
    #[cfg(all(windows, not(debug_assertions)))]
    // SAFETY: plain Win32 call with a constant argument; it fails harmlessly
    // when there is no parent console (e.g. started from the file explorer).
    unsafe {
        use windows_sys::Win32::System::Console::{ATTACH_PARENT_PROCESS, AttachConsole};
        AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

fn is_project(path: &std::path::Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case(polyflow_io::project::EXTENSION))
}

/// Exponential moving average, so displayed timings are readable.
fn smooth(value: &mut f32, sample: f32) {
    *value = if *value == 0.0 {
        sample
    } else {
        *value * 0.9 + sample * 0.1
    };
}

impl AppState {
    /// Why baking is not possible right now, if it is not. `uv_faces`: the
    /// retopology's faces with UVs.
    fn bake_blocker(&self, uv_faces: usize) -> Option<&'static str> {
        if self.bake_job.is_some() {
            Some("A bake is running.")
        } else if self.loading.is_some() {
            Some("The sculpt is still loading.")
        } else if self.mesh.triangle_count() == 0 {
            Some("Load a sculpt to bake from.")
        } else if self.retopo.mesh().is_empty() {
            Some("There is no retopology to bake onto.")
        } else if uv_faces == 0 {
            Some("The retopology has no UVs: unwrap it elsewhere and import it.")
        } else if !(self.bake_state.tangent_normal
            || self.bake_state.object_normal
            || self.bake_state.ambient_occlusion)
        {
            Some("Choose at least one map.")
        } else {
            None
        }
    }

    fn start_bake(&mut self) {
        let uv_faces = self.retopo.stats().uv_faces;
        if let Some(reason) = self.bake_blocker(uv_faces) {
            self.ui.error = Some(reason.into());
            return;
        }
        let cage = self.bake_state.use_cage.then(|| self.retopo.cage());
        let low = polyflow_bake::BakeMesh::with_cage(self.retopo.mesh(), cage.as_ref());
        let settings = bake::settings(&self.bake_state, self.retopo.scene_radius);
        let format =
            polyflow_io::image::MapFormat::from_name(&self.bake_state.format).unwrap_or_default();
        let prefix = bake::output_prefix(
            &self.bake_state,
            self.project_path.as_deref(),
            self.reference_path.as_deref(),
        );
        if low.faces_without_uvs > 0 {
            self.tools.status = Some(format!(
                "{} faces without UVs are left out of the bake.",
                low.faces_without_uvs
            ));
        }
        match bake::start(
            self.proxy.clone(),
            low,
            Arc::clone(&self.mesh),
            Arc::clone(&self.bvh),
            settings,
            format,
            prefix,
            self.bake_state.cpu,
        ) {
            Ok(job) => self.bake_job = Some(job),
            Err(e) => self.ui.error = Some(format!("Could not start the bake: {e}")),
        }
        self.window.request_redraw();
    }

    fn bake_done(&mut self, outcome: bake::BakeOutcome) {
        self.bake_job = None;
        self.window.request_redraw();
        match outcome {
            bake::BakeOutcome::Done(report) => {
                log::info!("Baked in {:?} on {}", report.time, report.device);
                let folder = report
                    .files
                    .first()
                    .and_then(|f| f.parent())
                    .map(|d| d.display().to_string())
                    .unwrap_or_default();
                self.tools.status = Some(format!("Bake saved in {folder}"));
                if let Err(e) = self.upload_preview(&report) {
                    log::error!("Could not show the bake preview: {e}");
                }
                self.last_bake = Some(report);
            }
            bake::BakeOutcome::Cancelled => self.tools.status = Some("Bake cancelled".into()),
            bake::BakeOutcome::Failed(e) => self.ui.error = Some(e),
        }
    }

    /// Shows `report` in the viewport, lit with its maps.
    fn upload_preview(&mut self, report: &bake::BakeReport) -> anyhow::Result<()> {
        let vertices: Vec<polyflow_render::PreviewVertex> = report
            .low
            .corners
            .iter()
            .map(|c| polyflow_render::PreviewVertex {
                position: c.position.to_array(),
                normal: c.normal.to_array(),
                tangent: c.tangent.to_array(),
                uv: c.uv.to_array(),
            })
            .collect();
        fn map(m: &Option<polyflow_bake::Map>) -> Option<polyflow_render::PreviewMap<'_>> {
            m.as_ref().map(|m| polyflow_render::PreviewMap {
                width: m.width,
                height: m.height,
                channels: m.channels,
                data: &m.data,
            })
        }
        self.renderer
            .set_preview(Some(&polyflow_render::PreviewData {
                vertices: &vertices,
                normal_map: map(&report.baked.tangent_normal),
                directx: report.directx,
                occlusion: map(&report.baked.ambient_occlusion),
            }))?;
        self.preview_mode = Some(polyflow_render::PreviewMode::Lit);
        Ok(())
    }

    fn choose_bake_output(&mut self) {
        let current = bake::output_prefix(
            &self.bake_state,
            self.project_path.as_deref(),
            self.reference_path.as_deref(),
        );
        let folder = current.parent().map(Path::to_path_buf);
        let name = current
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "bake".into());
        self.ask(
            move |parent| {
                let mut dialog = rfd::FileDialog::new()
                    .set_title("Bake output: maps are named <name>_normal and so on")
                    .set_file_name(name)
                    .set_parent(parent);
                if let Some(folder) = folder.filter(|f| f.is_dir()) {
                    dialog = dialog.set_directory(folder);
                }
                dialog.save_file()
            },
            |state, picked| {
                if let Some(path) = picked {
                    // The file picked is a name, not a file to write.
                    let prefix = path.with_extension("");
                    state.bake_state.output = Some(prefix.display().to_string());
                }
            },
        );
    }
}
