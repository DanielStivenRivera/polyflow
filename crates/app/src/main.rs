//! Polyflow desktop application.

mod bench;
mod camera_control;
mod jobs;
mod options;
mod ui;

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use polyflow_core::camera::OrbitCamera;
use polyflow_core::glam::{Vec2, Vec3};
use polyflow_core::mesh::TriMesh;
use polyflow_render::{FrameInput, MeshView, Renderer, UiFrame, Viewport};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::bench::Bench;
use crate::camera_control::CameraController;
use crate::jobs::{Job, Prepared};
use crate::options::Options;
use crate::ui::{Actions, MeshInfo, Stats, UiState};

/// Pixels of a high-resolution wheel/touchpad delta that count as one wheel line.
const PIXELS_PER_SCROLL_LINE: f32 = 50.0;
/// Linear RGB.
const BACKGROUND: Vec3 = Vec3::new(0.018, 0.020, 0.024);

/// Events sent to the event loop from other threads.
#[derive(Debug)]
enum UserEvent {
    MeshReady(Box<Result<Prepared, String>>),
}

fn main() -> anyhow::Result<()> {
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
            Ok(state) => self.state = Some(state),
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
    /// CPU copy of the reference mesh, for bounds now and spatial queries later.
    mesh: TriMesh,
    mesh_info: MeshInfo,
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

        let mut state = Self {
            renderer,
            window,
            proxy,
            egui_state,
            camera: OrbitCamera::default(),
            controller: CameraController::default(),
            mesh: TriMesh::default(),
            mesh_info: MeshInfo::empty(),
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
            ui,
        };
        let first_job = match &options.mesh {
            Some(path) => Job::Open(path.clone()),
            None => Job::Torus {
                rings: state.ui.torus_rings,
            },
        };
        state.start_job(first_job);
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
            WindowEvent::DroppedFile(path) => self.start_job(Job::Open(path)),
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
            "Showing {}: {} vertices, {} triangles, {} meshlets (meshlets built in {:?}, uploaded in {:?})",
            info.name,
            info.vertices,
            info.triangles,
            info.meshlets,
            info.meshlet_build,
            info.upload,
        );
        self.window.set_title(&format!("{} — Polyflow", info.name));
        self.mesh_info = info;
        self.mesh = prepared.mesh;
        self.frame_mesh();
        if let Some(factor) = self.initial_zoom.take() {
            self.camera.zoom(1.0 / factor);
        }
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
            self.start_job(Job::Open(path));
        }
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

        let stats = Stats {
            gpu: self.renderer.device_name(),
            present_mode: self.renderer.present_mode_name(),
            mesh: &self.mesh_info,
            loading: self.loading.as_deref(),
            culling: self.renderer.cull_stats(),
            cpu_ms: self.cpu_ms,
            interval_ms: self.interval_ms,
            gpu_ms: self.gpu_ms,
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
            self.start_job(Job::Torus {
                rings: self.ui.torus_rings,
            });
        }
        if actions.frame_mesh {
            self.frame_mesh();
        }
        self.renderer.set_vsync(self.ui.vsync);

        let [r, g, b] = self.ui.mesh_color;
        self.renderer.render(FrameInput {
            viewport: self.viewport,
            view: MeshView {
                view_proj: self.camera.view_projection(self.viewport_aspect()),
                eye: self.camera.eye(),
                color: Vec3::new(r, g, b),
                backface_culling: self.ui.backface_culling,
                show_meshlets: self.ui.show_meshlets,
            },
            clear_color: BACKGROUND,
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
            self.next_repaint = Instant::now().checked_add(repaint_delay);
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

/// Exponential moving average, so displayed timings are readable.
fn smooth(value: &mut f32, sample: f32) {
    *value = if *value == 0.0 {
        sample
    } else {
        *value * 0.9 + sample * 0.1
    };
}
