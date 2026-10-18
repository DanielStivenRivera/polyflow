//! The menu bar and side panel.

use std::time::Duration;

/// Description of the mesh currently displayed.
#[derive(Debug, Clone)]
pub struct MeshInfo {
    pub name: String,
    pub vertices: usize,
    pub triangles: usize,
    /// Present for meshes loaded from a file.
    pub file: Option<FileInfo>,
}

#[derive(Debug, Clone)]
pub struct FileInfo {
    pub bytes: u64,
    pub skipped_faces: usize,
    pub read: Duration,
    pub parse: Duration,
    pub normals: Duration,
    pub upload: Duration,
}

/// Read-only information shown in the panel.
#[derive(Debug)]
pub struct Stats<'a> {
    pub gpu: &'a str,
    pub present_mode: &'a str,
    pub mesh: &'a MeshInfo,
    /// Name of the file being loaded, if any.
    pub loading: Option<&'a str>,
    /// Smoothed time spent on the CPU per frame, in milliseconds.
    pub cpu_ms: f32,
    /// Smoothed time between frames, in milliseconds; only meaningful when
    /// redrawing continuously.
    pub interval_ms: f32,
}

/// Things the user asked for this frame.
#[derive(Debug, Default)]
pub struct Actions {
    pub open_file: bool,
    pub regenerate_mesh: bool,
    pub frame_mesh: bool,
    pub quit: bool,
}

#[derive(Debug)]
pub struct UiState {
    /// Segments around the torus; the tube gets half as many.
    pub torus_rings: u32,
    /// Linear RGB.
    pub mesh_color: [f32; 3],
    pub vsync: bool,
    pub continuous: bool,
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
            error: None,
        }
    }
}

impl UiState {
    pub fn torus_sides(&self) -> u32 {
        (self.torus_rings / 2).max(3)
    }

    /// Draws the UI. Returns the requested actions and the rect left for the
    /// 3D viewport, in points.
    pub fn show(&mut self, ui: &mut egui::Ui, stats: &Stats<'_>) -> (Actions, egui::Rect) {
        let mut actions = Actions::default();

        egui::Panel::top("menu_bar").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.add(egui::Button::new("Open…").shortcut_text("Ctrl+O")).clicked() {
                        actions.open_file = true;
                    }
                    ui.separator();
                    if ui.add(egui::Button::new("Quit").shortcut_text("Esc")).clicked() {
                        actions.quit = true;
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.add(egui::Button::new("Frame mesh").shortcut_text("F")).clicked() {
                        actions.frame_mesh = true;
                    }
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
                    self.side_panel(ui, stats, &mut actions);
                });
            });

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
            if let Some(file) = &mesh.file {
                row(ui, "File size", &format_bytes(file.bytes));
                row(ui, "Read", &format_duration(file.read));
                row(ui, "Parse", &format_duration(file.parse));
                row(ui, "Normals", &format_duration(file.normals));
                row(ui, "GPU upload", &format_duration(file.upload));
                if file.skipped_faces > 0 {
                    row(ui, "Skipped faces", &group_thousands(file.skipped_faces));
                }
            }
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
                egui::Slider::new(&mut self.torus_rings, 16..=4096)
                    .logarithmic(true)
                    .text("Rings"),
            );
            let pending = 2 * self.torus_rings as usize * self.torus_sides() as usize;
            if ui
                .button(format!("Generate {} triangles", group_thousands(pending)))
                .clicked()
            {
                actions.regenerate_mesh = true;
            }
        });

        ui.separator();
        ui.collapsing("Performance", |ui| {
            egui::Grid::new("performance")
                .num_columns(2)
                .show(ui, |ui| {
                    row(ui, "GPU", stats.gpu);
                    row(ui, "Present mode", stats.present_mode);
                    row(ui, "CPU frame", &format!("{:.2} ms", stats.cpu_ms));
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
                        ("Open", "Ctrl+O · drop a file"),
                    ] {
                        row(ui, action, binding);
                    }
                });
        });
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
