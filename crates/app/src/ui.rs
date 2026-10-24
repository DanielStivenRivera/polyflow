//! The menu bar and side panel.

use std::time::Duration;

use polyflow_core::glam::Vec2;

use crate::retopo::symmetry::Axis;
use crate::retopo::tools::{ToolKind, ToolSettings};

/// Description of the mesh currently displayed.
#[derive(Debug, Clone)]
pub struct MeshInfo {
    pub name: String,
    pub vertices: usize,
    pub triangles: usize,
    pub meshlets: usize,
    pub source: MeshSource,
    pub meshlet_build: Duration,
    pub bvh_build: Duration,
    pub bvh_bytes: usize,
    pub upload: Duration,
}

impl MeshInfo {
    pub fn empty() -> Self {
        Self {
            name: "None".into(),
            vertices: 0,
            triangles: 0,
            meshlets: 0,
            source: MeshSource::Generated {
                time: Duration::ZERO,
            },
            meshlet_build: Duration::ZERO,
            bvh_build: Duration::ZERO,
            bvh_bytes: 0,
            upload: Duration::ZERO,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MeshSource {
    File(FileInfo),
    Generated { time: Duration },
}

#[derive(Debug, Clone)]
pub struct FileInfo {
    pub bytes: u64,
    pub skipped_faces: usize,
    pub read: Duration,
    pub parse: Duration,
    pub normals: Duration,
}

/// Read-only information shown in the panel.
#[derive(Debug)]
pub struct Stats<'a> {
    pub gpu: &'a str,
    pub present_mode: &'a str,
    pub mesh: &'a MeshInfo,
    /// Description of the mesh being prepared, if any.
    pub loading: Option<&'a str>,
    /// Culling results of the last completed frame.
    pub culling: Option<polyflow_render::CullStats>,
    /// Surface under the cursor.
    pub hover: Option<&'a Hover>,
    /// Smoothed time spent on the CPU per frame, in milliseconds.
    pub cpu_ms: f32,
    /// Smoothed time between frames, in milliseconds; only meaningful when
    /// redrawing continuously.
    pub interval_ms: f32,
    /// Smoothed GPU time per frame, in milliseconds, if measurable.
    pub gpu_ms: Option<f32>,
    /// Relax brush outline (center and radius, physical pixels).
    pub brush: Option<(Vec2, f32)>,
    /// Whether the pointer is over the 3D viewport (for the cursor shape).
    pub cursor_in_viewport: bool,
}

/// The point of the reference mesh under the cursor.
#[derive(Debug, Clone, Copy)]
pub struct Hover {
    pub point: polyflow_core::glam::Vec3,
    pub triangle: u32,
    /// Time taken by the raycast.
    pub query: Duration,
}

/// Things the user asked for this frame.
#[derive(Debug, Default)]
pub struct Actions {
    pub open_file: bool,
    pub regenerate_mesh: bool,
    pub frame_mesh: bool,
    pub quit: bool,
    pub undo: bool,
    pub redo: bool,
    pub relax_all: bool,
    pub clear_retopo: bool,
    pub new_project: bool,
    pub open_project: bool,
    pub save_project: bool,
    pub save_project_as: bool,
    pub import_retopo: bool,
    pub export_retopo: bool,
}

/// Retopology state the UI shows and edits.
#[derive(Debug)]
pub struct RetopoUi<'a> {
    pub tool: &'a mut ToolKind,
    pub settings: &'a mut ToolSettings,
    pub symmetry: &'a mut Option<Axis>,
    pub x_ray: &'a mut bool,
    pub surface_offset: &'a mut f32,
    pub uv_checker: &'a mut bool,
    pub status: &'a mut Option<String>,
    pub undo_name: Option<&'a str>,
    pub redo_name: Option<&'a str>,
    pub vertices: usize,
    pub faces: usize,
    /// Triangles, quads and other polygons.
    pub face_kinds: [usize; 3],
    /// Faces with UVs on every corner.
    pub uv_faces: usize,
}

#[derive(Debug)]
pub struct UiState {
    /// Segments around the torus; the tube gets half as many.
    pub torus_rings: u32,
    /// Linear RGB.
    pub mesh_color: [f32; 3],
    pub vsync: bool,
    pub continuous: bool,
    pub backface_culling: bool,
    pub show_meshlets: bool,
    /// Zoom towards the surface under the cursor instead of the view center.
    pub zoom_to_cursor: bool,
    /// Orbit and pan at the depth of the surface under the cursor.
    pub auto_depth: bool,
    /// Last error to show the user, until dismissed.
    pub error: Option<String>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            torus_rings: 256,
            mesh_color: [0.48, 0.43, 0.39],
            vsync: true,
            continuous: false,
            backface_culling: true,
            show_meshlets: false,
            zoom_to_cursor: true,
            auto_depth: true,
            error: None,
        }
    }
}

impl UiState {
    /// Draws the UI. Returns the requested actions and the rect left for the
    /// 3D viewport, in points.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        stats: &Stats<'_>,
        retopo: &mut RetopoUi<'_>,
    ) -> (Actions, egui::Rect) {
        let mut actions = Actions::default();

        egui::Panel::top("menu_bar").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    let item = |ui: &mut egui::Ui, label: &str, shortcut: &str, flag: &mut bool| {
                        if ui.add(egui::Button::new(label).shortcut_text(shortcut)).clicked() {
                            *flag = true;
                        }
                    };
                    item(ui, "New project", "Ctrl+N", &mut actions.new_project);
                    item(ui, "Open project…", "Ctrl+O", &mut actions.open_project);
                    item(ui, "Save project", "Ctrl+S", &mut actions.save_project);
                    item(ui, "Save project as…", "Ctrl+Shift+S", &mut actions.save_project_as);
                    ui.separator();
                    item(ui, "Load reference mesh…", "Ctrl+L", &mut actions.open_file);
                    ui.separator();
                    item(ui, "Import retopology OBJ…", "Ctrl+I", &mut actions.import_retopo);
                    item(ui, "Export retopology OBJ…", "Ctrl+E", &mut actions.export_retopo);
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        actions.quit = true;
                    }
                });
                ui.menu_button("Edit", |ui| {
                    let undo = retopo.undo_name.map_or("Undo".to_owned(), |n| format!("Undo {n}"));
                    let button = egui::Button::new(undo).shortcut_text("Ctrl+Z");
                    if ui.add_enabled(retopo.undo_name.is_some(), button).clicked() {
                        actions.undo = true;
                    }
                    let redo = retopo.redo_name.map_or("Redo".to_owned(), |n| format!("Redo {n}"));
                    let button = egui::Button::new(redo).shortcut_text("Ctrl+Y");
                    if ui.add_enabled(retopo.redo_name.is_some(), button).clicked() {
                        actions.redo = true;
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.add(egui::Button::new("Frame mesh").shortcut_text("F")).clicked() {
                        actions.frame_mesh = true;
                    }
                    ui.checkbox(&mut self.backface_culling, "Backface culling").on_hover_text(
                        "Skip back faces. Turn off to see inside open meshes (back faces are tinted).",
                    );
                    ui.checkbox(&mut self.show_meshlets, "Show meshlets");
                    ui.separator();
                    ui.checkbox(&mut self.vsync, "VSync");
                    ui.checkbox(&mut self.continuous, "Redraw continuously")
                        .on_hover_text("Render every frame instead of only on changes, to measure performance.");
                });
            });
        });

        egui::Panel::left("side_panel")
            .resizable(false)
            .exact_size(260.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    retopo_panel(ui, retopo, &mut actions);
                    ui.separator();
                    self.side_panel(ui, stats, &mut actions);
                });
            });

        let ppp = ui.ctx().pixels_per_point();
        if let Some((center, radius)) = stats.brush {
            let painter = ui.ctx().layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("brush"),
            ));
            let center = egui::pos2(center.x / ppp, center.y / ppp);
            painter.circle_stroke(
                center,
                radius / ppp,
                egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 214, 64)),
            );
        }
        if stats.cursor_in_viewport {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        }

        (actions, ui.available_rect_before_wrap())
    }

    fn side_panel(&mut self, ui: &mut egui::Ui, stats: &Stats<'_>, actions: &mut Actions) {
        if let Some(error) = &self.error {
            let mut dismiss = false;
            egui::Frame::group(ui.style())
                .fill(ui.visuals().error_fg_color.gamma_multiply(0.15))
                .show(ui, |ui| {
                    ui.colored_label(ui.visuals().error_fg_color, error);
                    dismiss = ui.small_button("Dismiss").clicked();
                });
            if dismiss {
                self.error = None;
            }
            ui.add_space(4.0);
        }

        if let Some(name) = stats.loading {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!("Loading {name}…"));
            });
            ui.separator();
        }

        ui.strong("Reference mesh");
        let mesh = stats.mesh;
        egui::Grid::new("mesh_info").num_columns(2).show(ui, |ui| {
            row(ui, "Name", &mesh.name);
            row(ui, "Vertices", &group_thousands(mesh.vertices));
            row(ui, "Triangles", &group_thousands(mesh.triangles));
            row(ui, "Meshlets", &group_thousands(mesh.meshlets));
        });
        ui.collapsing("Load times", |ui| {
            egui::Grid::new("load_times").num_columns(2).show(ui, |ui| {
                match &mesh.source {
                    MeshSource::File(file) => {
                        row(ui, "File size", &format_bytes(file.bytes));
                        row(ui, "Read", &format_duration(file.read));
                        row(ui, "Parse", &format_duration(file.parse));
                        row(ui, "Normals", &format_duration(file.normals));
                        if file.skipped_faces > 0 {
                            row(ui, "Skipped faces", &group_thousands(file.skipped_faces));
                        }
                    }
                    MeshSource::Generated { time } => row(ui, "Generate", &format_duration(*time)),
                }
                row(ui, "Meshlets", &format_duration(mesh.meshlet_build));
                row(
                    ui,
                    "BVH",
                    &format!(
                        "{} ({})",
                        format_duration(mesh.bvh_build),
                        format_bytes(mesh.bvh_bytes as u64)
                    ),
                );
                row(ui, "GPU upload", &format_duration(mesh.upload));
            });
        });
        ui.horizontal(|ui| {
            if ui.button("Open…").clicked() {
                actions.open_file = true;
            }
            if ui.button("Frame (F)").clicked() {
                actions.frame_mesh = true;
            }
        });
        ui.horizontal(|ui| {
            ui.color_edit_button_rgb(&mut self.mesh_color);
            ui.label("Color");
        });

        ui.separator();
        ui.collapsing("Test mesh", |ui| {
            ui.add(
                egui::Slider::new(&mut self.torus_rings, 16..=8192)
                    .logarithmic(true)
                    .text("Rings"),
            );
            let pending =
                2 * self.torus_rings as usize * crate::jobs::torus_sides(self.torus_rings) as usize;
            if ui
                .button(format!("Generate {} triangles", group_thousands(pending)))
                .clicked()
            {
                actions.regenerate_mesh = true;
            }
        });

        ui.separator();
        ui.strong("Viewport");
        ui.checkbox(&mut self.backface_culling, "Backface culling")
            .on_hover_text(
                "Skip back faces. Turn off to see inside open meshes (back faces are tinted).",
            );
        ui.checkbox(&mut self.show_meshlets, "Show meshlets");
        ui.checkbox(&mut self.zoom_to_cursor, "Zoom to cursor");
        ui.checkbox(&mut self.auto_depth, "Auto depth")
            .on_hover_text("Orbit and pan around the depth of the surface under the cursor.");

        ui.separator();
        ui.strong("Under the cursor");
        match stats.hover {
            Some(hover) => {
                egui::Grid::new("hover").num_columns(2).show(ui, |ui| {
                    let p = hover.point;
                    row(
                        ui,
                        "Position",
                        &format!("{:.4}, {:.4}, {:.4}", p.x, p.y, p.z),
                    );
                    row(ui, "Triangle", &group_thousands(hover.triangle as usize));
                    row(
                        ui,
                        "Raycast",
                        &format!("{:.1} µs", hover.query.as_secs_f64() * 1e6),
                    );
                });
            }
            None => {
                ui.weak("Nothing");
            }
        }

        ui.separator();
        ui.collapsing("Performance", |ui| {
            egui::Grid::new("performance")
                .num_columns(2)
                .show(ui, |ui| {
                    row(ui, "GPU", stats.gpu);
                    row(ui, "Present mode", stats.present_mode);
                    row(ui, "CPU frame", &format!("{:.2} ms", stats.cpu_ms));
                    if let Some(gpu_ms) = stats.gpu_ms {
                        row(ui, "GPU frame", &format!("{gpu_ms:.2} ms"));
                    }
                    if let Some(c) = stats.culling {
                        let percent =
                            |part: u32, whole: usize| 100.0 * part as f64 / whole.max(1) as f64;
                        row(
                            ui,
                            "Visible meshlets",
                            &format!(
                                "{} ({:.0}%)",
                                group_thousands(c.visible_meshlets as usize),
                                percent(c.visible_meshlets, c.total_meshlets as usize)
                            ),
                        );
                        row(
                            ui,
                            "Visible triangles",
                            &format!(
                                "{} ({:.0}%)",
                                group_thousands(c.visible_triangles as usize),
                                percent(c.visible_triangles, stats.mesh.triangles)
                            ),
                        );
                    }
                    if self.continuous && stats.interval_ms > 0.0 {
                        row(
                            ui,
                            "Frame rate",
                            &format!(
                                "{:.0} fps ({:.2} ms)",
                                1000.0 / stats.interval_ms,
                                stats.interval_ms
                            ),
                        );
                    }
                });
        });

        ui.collapsing("Navigation", |ui| {
            egui::Grid::new("navigation")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    for (action, binding) in [
                        ("Orbit", "MMB · Alt+LMB"),
                        ("Pan", "Shift+MMB · Alt+MMB"),
                        ("Zoom", "Wheel · Ctrl+MMB · Alt+RMB"),
                        ("Frame", "F"),
                        ("Open project", "Ctrl+O · drop a .pflow"),
                        ("Load sculpt", "Ctrl+L · drop an .obj"),
                        ("Undo / redo", "Ctrl+Z · Ctrl+Y"),
                        ("Save project", "Ctrl+S"),
                        ("Tools", "Q create · D draw · R relax"),
                        ("Symmetry X", "X"),
                        ("Cancel", "Esc · right click"),
                    ] {
                        row(ui, action, binding);
                    }
                });
        });
    }
}

fn retopo_panel(ui: &mut egui::Ui, retopo: &mut RetopoUi<'_>, actions: &mut Actions) {
    ui.strong("Retopology");
    ui.horizontal(|ui| {
        for tool in ToolKind::ALL {
            ui.selectable_value(retopo.tool, tool, tool.label())
                .on_hover_text(format!("{} ({})", tool_help(tool), tool.shortcut()));
        }
    });
    match retopo.tool {
        ToolKind::Create => {
            ui.weak("Click the surface to add vertices; 4 make a quad. Click the first vertex to close a polygon.");
        }
        ToolKind::Draw => {
            ui.add(
                egui::Slider::new(&mut retopo.settings.strip_width, 10.0..=200.0)
                    .text("Width (px)"),
            );
        }
        ToolKind::Relax => {
            ui.add(
                egui::Slider::new(&mut retopo.settings.brush_radius, 10.0..=400.0)
                    .text("Radius (px)"),
            );
            ui.add(
                egui::Slider::new(&mut retopo.settings.brush_strength, 0.05..=1.0).text("Strength"),
            );
            ui.checkbox(&mut retopo.settings.relax_borders, "Relax borders")
                .on_hover_text(
                    "Also smooth border vertices (along the border). Off keeps borders pinned.",
                );
        }
    }
    ui.horizontal(|ui| {
        ui.label("Symmetry");
        ui.selectable_value(retopo.symmetry, None, "Off");
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            ui.selectable_value(retopo.symmetry, Some(axis), axis.label());
        }
    });
    ui.horizontal(|ui| {
        ui.checkbox(retopo.x_ray, "X-ray")
            .on_hover_text("Show and pick the retopo mesh through the surface.");
        let mut percent = *retopo.surface_offset * 100.0;
        let slider = egui::Slider::new(&mut percent, 0.0..=5.0)
            .text("Offset %")
            .max_decimals(1);
        let hint = "Draw the retopo mesh this far in front of the surface, so its faces do not sink under it.";
        if ui.add(slider).on_hover_text(hint).changed() {
            *retopo.surface_offset = percent / 100.0;
        }
    });
    ui.checkbox(retopo.uv_checker, "UV checker").on_hover_text(
        "Shade faces with a checker from their UVs (red along U, green along V). \
         Faces without UVs show in magenta: they will not bake.",
    );
    ui.horizontal(|ui| {
        if ui
            .add_enabled(retopo.undo_name.is_some(), egui::Button::new("Undo"))
            .clicked()
        {
            actions.undo = true;
        }
        if ui
            .add_enabled(retopo.redo_name.is_some(), egui::Button::new("Redo"))
            .clicked()
        {
            actions.redo = true;
        }
        if ui
            .add_enabled(retopo.vertices > 0, egui::Button::new("Relax all"))
            .clicked()
        {
            actions.relax_all = true;
        }
    });
    egui::Grid::new("retopo_stats")
        .num_columns(2)
        .show(ui, |ui| {
            row(ui, "Vertices", &group_thousands(retopo.vertices));
            let [tris, quads, ngons] = retopo.face_kinds;
            row(
                ui,
                "Faces",
                &format!(
                    "{} ({} quads, {} tris, {} n-gons)",
                    group_thousands(retopo.faces),
                    quads,
                    tris,
                    ngons
                ),
            );
            let uvs = match retopo.uv_faces {
                0 => "None".to_owned(),
                n if n == retopo.faces => "All faces".to_owned(),
                n => format!(
                    "{} of {} faces",
                    group_thousands(n),
                    group_thousands(retopo.faces)
                ),
            };
            row(ui, "UVs", &uvs);
        });
    if retopo.vertices > 0 && ui.small_button("Clear retopology").clicked() {
        actions.clear_retopo = true;
    }
    if let Some(status) = retopo.status.clone() {
        ui.horizontal(|ui| {
            ui.colored_label(ui.visuals().warn_fg_color, status);
            if ui.small_button("×").clicked() {
                *retopo.status = None;
            }
        });
    }
}

fn tool_help(tool: ToolKind) -> &'static str {
    match tool {
        ToolKind::Create => {
            "Place vertices and build faces. Ctrl+drag moves a vertex, Delete removes what is under the cursor"
        }
        ToolKind::Draw => "Drag over the surface to draw a strip of quads",
        ToolKind::Relax => "Drag to smooth the retopo mesh, keeping it on the surface",
    }
}

fn row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.label(label);
    ui.label(value);
    ui.end_row();
}

fn group_thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn format_duration(d: Duration) -> String {
    let ms = d.as_secs_f64() * 1000.0;
    if ms >= 1000.0 {
        format!("{:.2} s", ms / 1000.0)
    } else {
        format!("{ms:.0} ms")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formatting() {
        assert_eq!(group_thousands(0), "0");
        assert_eq!(group_thousands(999), "999");
        assert_eq!(group_thousands(1000), "1,000");
        assert_eq!(group_thousands(16_777_216), "16,777,216");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(3 << 30), "3.0 GB");
        assert_eq!(format_duration(Duration::from_millis(42)), "42 ms");
        assert_eq!(format_duration(Duration::from_millis(2500)), "2.50 s");
    }
}
