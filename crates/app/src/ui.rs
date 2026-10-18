//! The side panel.

/// Read-only information shown in the panel.
#[derive(Debug)]
pub struct Stats<'a> {
    pub gpu: &'a str,
    pub present_mode: &'a str,
    pub triangles: usize,
    /// Smoothed time spent on the CPU per frame, in milliseconds.
    pub cpu_ms: f32,
    /// Smoothed time between frames, in milliseconds; only meaningful when
    /// redrawing continuously.
    pub interval_ms: f32,
}

/// Things the user asked for this frame.
#[derive(Debug, Default)]
pub struct Actions {
    pub regenerate_mesh: bool,
    pub frame_mesh: bool,
}

#[derive(Debug)]
pub struct UiState {
    /// Segments around the torus; the tube gets half as many.
    pub torus_rings: u32,
    /// Linear RGB.
    pub mesh_color: [f32; 3],
    pub vsync: bool,
    pub continuous: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            torus_rings: 256,
            mesh_color: [0.48, 0.43, 0.39],
            vsync: true,
            continuous: false,
        }
    }
}

impl UiState {
    pub fn torus_sides(&self) -> u32 {
        (self.torus_rings / 2).max(3)
    }

    /// Draws the panel. Returns the requested actions and the rect left for
    /// the 3D viewport, in points.
    pub fn show(&mut self, ui: &mut egui::Ui, stats: &Stats<'_>) -> (Actions, egui::Rect) {
        let mut actions = Actions::default();
        egui::Panel::left("side_panel")
            .resizable(false)
            .exact_size(240.0)
            .show(ui, |ui| {
                ui.heading("Polyflow");
                ui.label(format!("GPU: {}", stats.gpu));
                ui.label(format!("Present mode: {}", stats.present_mode));
                ui.label(format!("CPU frame time: {:.2} ms", stats.cpu_ms));
                if self.continuous && stats.interval_ms > 0.0 {
                    ui.label(format!(
                        "{:.0} fps ({:.2} ms)",
                        1000.0 / stats.interval_ms,
                        stats.interval_ms
                    ));
                }

                ui.separator();
                ui.strong("Test mesh");
                ui.label(format!("{} triangles", group_thousands(stats.triangles)));
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
                ui.horizontal(|ui| {
                    ui.color_edit_button_rgb(&mut self.mesh_color);
                    ui.label("Color");
                });

                ui.separator();
                ui.strong("Viewport");
                ui.checkbox(&mut self.vsync, "VSync");
                ui.checkbox(&mut self.continuous, "Redraw continuously")
                    .on_hover_text(
                        "Render every frame instead of only on changes, to measure performance.",
                    );
                if ui.button("Frame mesh (F)").clicked() {
                    actions.frame_mesh = true;
                }

                ui.separator();
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
                            ] {
                                ui.label(action);
                                ui.label(binding);
                                ui.end_row();
                            }
                        });
                });
            });
        (actions, ui.available_rect_before_wrap())
    }
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

#[cfg(test)]
mod tests {
    #[test]
    fn group_thousands() {
        assert_eq!(super::group_thousands(0), "0");
        assert_eq!(super::group_thousands(999), "999");
        assert_eq!(super::group_thousands(1000), "1,000");
        assert_eq!(super::group_thousands(16_777_216), "16,777,216");
    }
}
