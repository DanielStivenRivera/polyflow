//! Polyflow desktop application.

mod camera_control;
mod ui;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use polyflow_core::camera::OrbitCamera;
use polyflow_core::glam::{Vec2, Vec3};
use polyflow_core::mesh::TriMesh;
use polyflow_core::primitives;
use polyflow_io::{LoadError, LoadedMesh};
use polyflow_render::{FrameInput, Renderer, UiFrame, Viewport};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::camera_control::CameraController;
use crate::ui::{Actions, FileInfo, MeshInfo, Stats, UiState};

/// Pixels of a high-resolution wheel/touchpad delta that count as one wheel line.
const PIXELS_PER_SCROLL_LINE: f32 = 50.0;
/// Linear RGB.
const BACKGROUND: Vec3 = Vec3::new(0.018, 0.020, 0.024);

/// Events sent to the event loop from other threads.
#[derive(Debug)]
enum UserEvent {
    MeshLoaded {
        path: PathBuf,
        result: Box<Result<LoadedMesh, LoadError>>,
    },
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let mut app = App {
        state: None,
        error: None,
        proxy: event_loop.create_proxy(),
        initial_file: std::env::args_os().nth(1).map(PathBuf::from),
    };
    event_loop.run_app(&mut app)?;
    app.error.map_or(Ok(()), Err)
}

struct App {
    state: Option<AppState>,
    error: Option<anyhow::Error>,
    proxy: EventLoopProxy<UserEvent>,
    /// File given on the command line, opened once the window exists.
    initial_file: Option<PathBuf>,
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
        match AppState::new(event_loop, self.proxy.clone()) {
            Ok(mut state) => {
                if let Some(path) = self.initial_file.take() {
                    state.start_loading(path);
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
            UserEvent::MeshLoaded { path, result } => state.mesh_loaded(&path, *result),
        };
        self.handle(event_loop, result);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = &mut self.state else { return };
        match state.next_repaint {
            Some(at) if at <= Instant::now() => {
                state.next_repaint = None;
                state.window.request_redraw();
                event_loop.set_control_flow(ControlFlow::Wait);
            }
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
    mesh: TriMesh,
    mesh_info: MeshInfo,
    /// File currently being loaded on a background thread.
    loading: Option<PathBuf>,
    /// Region of the window showing the 3D scene, in physical pixels.
    viewport: Viewport,
    next_repaint: Option<Instant>,
    last_frame: Option<Instant>,
    cpu_ms: f32,
    interval_ms: f32,
}

impl AppState {
    fn new(event_loop: &ActiveEventLoop, proxy: EventLoopProxy<UserEvent>) -> anyhow::Result<Self> {
        let attributes = Window::default_attributes()
            .with_title("Polyflow")
            .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 900.0));
        let window = Arc::new(event_loop.create_window(attributes)?);
        let size = window.inner_size();
        let ui = UiState::default();
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

        let mut state = Self {
            renderer,
            window,
            proxy,
            egui_state,
            camera: OrbitCamera::default(),
            controller: CameraController::default(),
            ui,
            mesh: TriMesh::default(),
            mesh_info: MeshInfo {
                name: String::new(),
                vertices: 0,
                triangles: 0,
                file: None,
            },
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
        };
        state.generate_test_mesh()?;
        Ok(state)
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        event: WindowEvent,
    ) -> anyhow::Result<()> {
        let response = self.egui_state.on_window_event(&self.window, &event);
        if response.repaint {
            self.window.request_redraw();
        }
        let viewport_height = self.viewport.height;

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.renderer.resize(size.width, size.height);
                self.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let actions = self.redraw()?;
                self.apply_post_frame_actions(event_loop, &actions);
            }
            WindowEvent::DroppedFile(path) => self.start_loading(path),
            WindowEvent::ModifiersChanged(modifiers) => {
                self.controller.modifiers = modifiers.state()
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let over_ui = self.pointer_over_ui();
                self.controller
                    .mouse_button(button, state == ElementState::Pressed, over_ui);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = Vec2::new(position.x as f32, position.y as f32);
                if self
                    .controller
                    .cursor_moved(position, &mut self.camera, viewport_height)
                {
                    self.window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !self.pointer_over_ui() => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / PIXELS_PER_SCROLL_LINE,
                };
                self.controller.scroll(lines, &mut self.camera);
                self.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !response.consumed && !event.repeat =>
            {
                let ctrl = self.controller.modifiers.control_key();
                match &event.logical_key {
                    Key::Character(c) if c.eq_ignore_ascii_case("o") && ctrl => {
                        self.open_file_dialog()
                    }
                    Key::Character(c) if c.eq_ignore_ascii_case("f") && !ctrl => {
                        self.frame_mesh();
                        self.window.request_redraw();
                    }
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    _ => {}
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Loads `path` on a background thread; the result arrives as a
    /// [`UserEvent::MeshLoaded`].
    fn start_loading(&mut self, path: PathBuf) {
        if let Some(current) = &self.loading {
            self.ui.error = Some(format!("Already loading {}", display_name(current)));
            self.window.request_redraw();
            return;
        }
        log::info!("Loading {}", path.display());
        self.loading = Some(path.clone());
        let proxy = self.proxy.clone();
        let spawned = std::thread::Builder::new()
            .name("mesh loader".into())
            .spawn(move || {
                let result = Box::new(polyflow_io::load_mesh(&path));
                // Fails only if the event loop is gone, i.e. the app is closing.
                let _ = proxy.send_event(UserEvent::MeshLoaded { path, result });
            });
        if let Err(e) = spawned {
            self.loading = None;
            self.ui.error = Some(format!("Could not start loading: {e}"));
        }
        self.window.request_redraw();
    }

    fn mesh_loaded(
        &mut self,
        path: &Path,
        result: Result<LoadedMesh, LoadError>,
    ) -> anyhow::Result<()> {
        self.loading = None;
        self.window.request_redraw();
        let loaded = match result {
            Ok(loaded) => loaded,
            Err(e) => {
                log::error!("Failed to load {}: {e}", path.display());
                self.ui.error = Some(format!("Could not open {}: {e}", display_name(path)));
                return Ok(());
            }
        };

        let start = Instant::now();
        self.renderer.set_mesh(&loaded.mesh)?;
        let upload = start.elapsed();

        let file = FileInfo {
            bytes: loaded.file_bytes,
            skipped_faces: loaded.skipped_faces,
            read: loaded.read_time,
            parse: loaded.parse_time,
            normals: loaded.normals_time,
            upload,
        };
        log::info!(
            "Loaded {}: {} vertices, {} triangles (read {:?}, parse {:?}, normals {:?}, upload {:?})",
            path.display(),
            loaded.mesh.vertex_count(),
            loaded.mesh.triangle_count(),
            file.read,
            file.parse,
            file.normals,
            file.upload,
        );
        self.mesh_info = MeshInfo {
            name: display_name(path),
            vertices: loaded.mesh.vertex_count(),
            triangles: loaded.mesh.triangle_count(),
            file: Some(file),
        };
        self.mesh = loaded.mesh;
        self.frame_mesh();
        self.window
            .set_title(&format!("{} — Polyflow", self.mesh_info.name));
        Ok(())
    }

    fn open_file_dialog(&mut self) {
        let picked = rfd::FileDialog::new()
            .set_title("Open reference mesh")
            .add_filter("Meshes", &["obj"])
            .add_filter("All files", &["*"])
            .set_parent(self.window.as_ref())
            .pick_file();
        if let Some(path) = picked {
            self.start_loading(path);
        }
    }

    fn generate_test_mesh(&mut self) -> anyhow::Result<()> {
        let start = Instant::now();
        let mesh = primitives::torus(1.0, 0.35, self.ui.torus_rings, self.ui.torus_sides());
        self.renderer.set_mesh(&mesh)?;
        log::info!(
            "Test mesh: {} triangles in {:?}",
            mesh.triangle_count(),
            start.elapsed()
        );
        self.mesh_info = MeshInfo {
            name: "Test torus".into(),
            vertices: mesh.vertex_count(),
            triangles: mesh.triangle_count(),
            file: None,
        };
        self.mesh = mesh;
        self.frame_mesh();
        self.window.set_title("Polyflow");
        Ok(())
    }

    fn frame_mesh(&mut self) {
        let aspect = self.viewport_aspect();
        self.camera.frame(&self.mesh.bounds(), aspect);
    }

    /// Runs the UI and renders a frame. Actions that block (dialogs) or end
    /// the app are returned to run after the frame is presented.
    fn redraw(&mut self) -> anyhow::Result<Actions> {
        let start = Instant::now();
        if let Some(last) = self.last_frame.replace(start) {
            smooth(&mut self.interval_ms, (start - last).as_secs_f32() * 1000.0);
        }

        let loading_name = self.loading.as_deref().map(display_name);
        let stats = Stats {
            gpu: self.renderer.device_name(),
            present_mode: self.renderer.present_mode_name(),
            mesh: &self.mesh_info,
            loading: loading_name.as_deref(),
            cpu_ms: self.cpu_ms,
            interval_ms: self.interval_ms,
        };
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let mut actions = Actions::default();
        let mut viewport_rect = egui::Rect::NOTHING;
        let ui_state = &mut self.ui;
        let output = self.egui_state.egui_ctx().run_ui(raw_input, |ui| {
            (actions, viewport_rect) = ui_state.show(ui, &stats)
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
            self.generate_test_mesh()?;
        }
        if actions.frame_mesh {
            self.frame_mesh();
        }
        self.renderer.set_vsync(self.ui.vsync);

        let aspect = self.viewport_aspect();
        let [r, g, b] = self.ui.mesh_color;
        self.renderer.render(FrameInput {
            viewport: self.viewport,
            view_proj: self.camera.view_projection(aspect),
            eye: self.camera.eye(),
            clear_color: BACKGROUND,
            mesh_color: Vec3::new(r, g, b),
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
        if self.ui.continuous
            || repaint_delay.is_zero()
            || actions.regenerate_mesh
            || actions.frame_mesh
        {
            self.window.request_redraw();
        } else {
            self.next_repaint = Instant::now().checked_add(repaint_delay);
            // Frame intervals are only meaningful while redrawing back to back.
            self.last_frame = None;
        }
        smooth(&mut self.cpu_ms, start.elapsed().as_secs_f32() * 1000.0);
        Ok(actions)
    }

    fn apply_post_frame_actions(&mut self, event_loop: &ActiveEventLoop, actions: &Actions) {
        if actions.quit {
            event_loop.exit();
        } else if actions.open_file {
            self.open_file_dialog();
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

fn display_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

/// Exponential moving average, so displayed timings are readable.
fn smooth(value: &mut f32, sample: f32) {
    *value = if *value == 0.0 {
        sample
    } else {
        *value * 0.9 + sample * 0.1
    };
}
