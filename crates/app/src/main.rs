//! Polyflow desktop application.

mod camera_control;
mod ui;

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use polyflow_core::camera::OrbitCamera;
use polyflow_core::glam::{Vec2, Vec3};
use polyflow_core::primitives::{self, TriMesh};
use polyflow_render::{FrameInput, Renderer, UiFrame, Viewport};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::camera_control::CameraController;
use crate::ui::{Stats, UiState};

/// Pixels of a high-resolution wheel/touchpad delta that count as one wheel line.
const PIXELS_PER_SCROLL_LINE: f32 = 50.0;
/// Linear RGB.
const BACKGROUND: Vec3 = Vec3::new(0.018, 0.020, 0.024);

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let event_loop = EventLoop::new()?;
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    app.error.map_or(Ok(()), Err)
}

#[derive(Default)]
struct App {
    state: Option<AppState>,
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            match AppState::new(event_loop) {
                Ok(state) => self.state = Some(state),
                Err(e) => {
                    self.error = Some(e);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        if let Err(e) = state.window_event(event_loop, event) {
            self.error = Some(e);
            event_loop.exit();
        }
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
    egui_state: egui_winit::State,
    camera: OrbitCamera,
    controller: CameraController,
    ui: UiState,
    mesh: TriMesh,
    /// Region of the window showing the 3D scene, in physical pixels.
    viewport: Viewport,
    next_repaint: Option<Instant>,
    last_frame: Option<Instant>,
    cpu_ms: f32,
    interval_ms: f32,
}

impl AppState {
    fn new(event_loop: &ActiveEventLoop) -> anyhow::Result<Self> {
        let attributes = Window::default_attributes()
            .with_title("Polyflow")
            .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 900.0));
        let window = Arc::new(event_loop.create_window(attributes)?);
        let size = window.inner_size();
        let ui = UiState::default();
        let mut renderer = Renderer::new(window.as_ref(), size.width, size.height, ui.vsync)
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

        let mesh = primitives::torus(1.0, 0.35, ui.torus_rings, ui.torus_sides());
        renderer.set_mesh(&mesh)?;
        let mut camera = OrbitCamera::default();
        camera.frame(
            &mesh.bounds(),
            size.width.max(1) as f32 / size.height.max(1) as f32,
        );

        Ok(Self {
            renderer,
            window,
            egui_state,
            camera,
            controller: CameraController::default(),
            ui,
            viewport: Viewport {
                width: size.width as f32,
                height: size.height as f32,
                ..Default::default()
            },
            mesh,
            next_repaint: None,
            last_frame: None,
            cpu_ms: 0.0,
            interval_ms: 0.0,
        })
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
            WindowEvent::RedrawRequested => self.redraw()?,
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
                if let Key::Character(c) = &event.logical_key
                    && c.eq_ignore_ascii_case("f")
                {
                    self.camera
                        .frame(&self.mesh.bounds(), self.viewport_aspect());
                    self.window.request_redraw();
                }
                if event.logical_key == Key::Named(NamedKey::Escape) {
                    event_loop.exit();
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn redraw(&mut self) -> anyhow::Result<()> {
        let start = Instant::now();
        if let Some(last) = self.last_frame.replace(start) {
            smooth(&mut self.interval_ms, (start - last).as_secs_f32() * 1000.0);
        }

        let stats = Stats {
            gpu: self.renderer.device_name(),
            present_mode: self.renderer.present_mode_name(),
            triangles: self.mesh.triangle_count(),
            cpu_ms: self.cpu_ms,
            interval_ms: self.interval_ms,
        };
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let mut actions = Default::default();
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

        if actions.regenerate_mesh {
            let generated = Instant::now();
            self.mesh = primitives::torus(1.0, 0.35, self.ui.torus_rings, self.ui.torus_sides());
            let uploaded = Instant::now();
            self.renderer.set_mesh(&self.mesh)?;
            log::info!(
                "Test mesh: {} triangles, generated in {:.0} ms, uploaded in {:.0} ms",
                self.mesh.triangle_count(),
                (uploaded - generated).as_secs_f32() * 1000.0,
                uploaded.elapsed().as_secs_f32() * 1000.0
            );
        }
        if actions.frame_mesh {
            self.camera
                .frame(&self.mesh.bounds(), self.viewport_aspect());
        }
        self.renderer.set_vsync(self.ui.vsync);

        let ppp = output.pixels_per_point;
        self.viewport = Viewport {
            x: viewport_rect.min.x * ppp,
            y: viewport_rect.min.y * ppp,
            width: viewport_rect.width().max(0.0) * ppp,
            height: viewport_rect.height().max(0.0) * ppp,
        };
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
        Ok(())
    }
}

impl AppState {
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
