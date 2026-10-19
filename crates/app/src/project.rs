//! Project files, unsaved-change tracking, autosave and crash recovery.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use polyflow_core::camera::OrbitCamera;
use polyflow_core::glam::Vec3;
use polyflow_core::halfedge::HalfEdgeMesh;
use polyflow_io::project::{self, CameraState, ProjectSettings, ReferencePath, ToolState};

use crate::AppState;
use crate::jobs::{self, Job};
use crate::retopo::symmetry::Axis;
use crate::retopo::tools::ToolSettings;

/// How often unsaved work is written to the recovery folder.
pub const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(120);
const AUTOSAVE_PREFIX: &str = "autosave-";

/// Autosave bookkeeping.
#[derive(Debug)]
pub struct Autosave {
    pub enabled: bool,
    /// When the next autosave is due (only while there are unsaved changes).
    pub next: Instant,
    /// What the last autosave captured, to skip identical ones.
    saved_state: Option<(u64, Option<PathBuf>)>,
}

impl Autosave {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            next: Instant::now() + autosave_interval(),
            saved_state: None,
        }
    }
}

/// [`AUTOSAVE_INTERVAL`], or `POLYFLOW_AUTOSAVE_SECS` for testing.
fn autosave_interval() -> Duration {
    std::env::var("POLYFLOW_AUTOSAVE_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&secs| secs > 0)
        .map_or(AUTOSAVE_INTERVAL, Duration::from_secs)
}

/// Folder holding autosaves: the platform's local data folder.
fn recovery_dir() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join("Polyflow").join("recovery"))
}

/// This process's autosave file.
fn autosave_path() -> Option<PathBuf> {
    recovery_dir().map(|d| {
        d.join(format!(
            "{AUTOSAVE_PREFIX}{}.{}",
            std::process::id(),
            project::EXTENSION
        ))
    })
}

/// Autosaves left by other (crashed) sessions, newest first.
fn stale_autosaves() -> Vec<(PathBuf, SystemTime)> {
    let Some(dir) = recovery_dir() else {
        return Vec::new();
    };
    let own = autosave_path();
    let mut found: Vec<(PathBuf, SystemTime)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            let ours = Some(&path) == own.as_ref();
            let is_autosave = name.starts_with(AUTOSAVE_PREFIX)
                && name.ends_with(&format!(".{}", project::EXTENSION));
            if !is_autosave || ours {
                return None;
            }
            Some((path.clone(), entry.metadata().ok()?.modified().ok()?))
        })
        .collect();
    found.sort_by_key(|&(_, modified)| std::cmp::Reverse(modified));
    found
}

fn describe_age(time: SystemTime) -> String {
    let minutes = time.elapsed().unwrap_or_default().as_secs() / 60;
    match minutes {
        0 => "less than a minute ago".into(),
        1 => "1 minute ago".into(),
        m if m < 120 => format!("{m} minutes ago"),
        m => format!("{} hours ago", m / 60),
    }
}

impl AppState {
    /// Whether there is work that would be lost on closing.
    pub(crate) fn has_unsaved_changes(&self) -> bool {
        let mesh_changed = self.retopo.mesh().version() != self.saved_version;
        // A different sculpt only matters once there is a project to save it in.
        let reference_changed = self.reference_path != self.saved_reference
            && (self.project_path.is_some() || !self.retopo.mesh().is_empty());
        mesh_changed || reference_changed
    }

    pub(crate) fn project_settings(&self, project_path: &Path) -> ProjectSettings {
        let c = &self.camera;
        let t = &self.tools.settings;
        ProjectSettings {
            reference: self
                .reference_path
                .as_deref()
                .map(|mesh| ReferencePath::new(mesh, project_path)),
            camera: Some(CameraState {
                target: c.target.to_array(),
                distance: c.distance,
                yaw: c.yaw,
                pitch: c.pitch,
                fov_y: c.fov_y,
            }),
            symmetry: self.retopo.symmetry_axis.map(|a| match a {
                Axis::X => project::Axis::X,
                Axis::Y => project::Axis::Y,
                Axis::Z => project::Axis::Z,
            }),
            x_ray: self.retopo.x_ray,
            surface_offset: self.retopo.surface_offset,
            tools: ToolState {
                strip_width: t.strip_width,
                brush_radius: t.brush_radius,
                brush_strength: t.brush_strength,
                relax_borders: t.relax_borders,
            },
        }
    }

    fn apply_project_settings(&mut self, settings: &ProjectSettings) {
        self.retopo.symmetry_axis = settings.symmetry.map(|a| match a {
            project::Axis::X => Axis::X,
            project::Axis::Y => Axis::Y,
            project::Axis::Z => Axis::Z,
        });
        self.retopo.x_ray = settings.x_ray;
        self.retopo.surface_offset = settings.surface_offset;
        let t = settings.tools;
        self.tools.settings = ToolSettings {
            strip_width: t.strip_width,
            brush_radius: t.brush_radius,
            brush_strength: t.brush_strength,
            relax_borders: t.relax_borders,
        };
    }

    /// Applies a saved camera, if any.
    pub(crate) fn apply_camera(&mut self, camera: &CameraState) {
        let near = self.camera.near;
        self.camera = OrbitCamera {
            target: Vec3::from_array(camera.target),
            distance: camera.distance.max(1e-6),
            yaw: camera.yaw,
            pitch: camera.pitch,
            fov_y: camera.fov_y.clamp(0.05, 3.0),
            near: near.min(camera.distance * 0.1).max(1e-7),
        };
    }

    // ---- Commands --------------------------------------------------------------

    /// Saves the project, asking for a file if there is none yet or `ask` is
    /// set. Returns whether it was saved.
    pub(crate) fn save_project(&mut self, ask: bool) -> bool {
        let path = match (&self.project_path, ask) {
            (Some(path), false) => path.clone(),
            _ => {
                let default_name = self
                    .project_path
                    .as_deref()
                    .or(self.reference_path.as_deref())
                    .and_then(|p| p.file_stem())
                    .map_or("untitled".to_owned(), |s| s.to_string_lossy().into_owned());
                let picked = rfd::FileDialog::new()
                    .set_title("Save project")
                    .add_filter("Polyflow project", &[project::EXTENSION])
                    .set_file_name(format!("{default_name}.{}", project::EXTENSION))
                    .set_parent(self.window.as_ref())
                    .save_file();
                match picked {
                    Some(path) => path.with_extension(project::EXTENSION),
                    None => return false,
                }
            }
        };
        let settings = self.project_settings(&path);
        match project::save(&path, &settings, self.retopo.mesh()) {
            Ok(()) => {
                log::info!("Saved project to {}", path.display());
                self.tools.status = Some(format!("Saved {}", jobs::display_name(&path)));
                self.saved_version = self.retopo.mesh().version();
                self.saved_reference = self.reference_path.clone();
                self.project_path = Some(path);
                self.remove_autosave();
                true
            }
            Err(e) => {
                self.ui.error = Some(format!("Could not save {}: {e}", path.display()));
                false
            }
        }
    }

    pub(crate) fn open_project_dialog(&mut self) {
        if self.has_unsaved_changes()
            && !self.confirm_discard("Opening a project replaces the current work.")
        {
            return;
        }
        let picked = rfd::FileDialog::new()
            .set_title("Open project")
            .add_filter("Polyflow project", &[project::EXTENSION])
            .set_parent(self.window.as_ref())
            .pick_file();
        if let Some(path) = picked {
            self.open_project(&path, false);
        }
    }

    /// Opens a project. A `recovered` autosave opens as unsaved, untitled work.
    pub(crate) fn open_project(&mut self, path: &Path, recovered: bool) {
        let project = match project::load(path) {
            Ok(project) => project,
            Err(e) => {
                self.ui.error = Some(format!("Could not open {}: {e}", jobs::display_name(path)));
                return;
            }
        };
        log::info!("Opened project {}", path.display());
        self.retopo.replace_mesh(project.retopo);
        self.tools.reset();
        self.apply_project_settings(&project.settings);
        self.project_path = (!recovered).then(|| path.to_path_buf());
        // A recovered autosave is unsaved work: make sure it reads as changed.
        self.saved_version = if recovered {
            u64::MAX
        } else {
            self.retopo.mesh().version()
        };
        self.tools.status = Some(match project.skipped_faces {
            0 if recovered => "Recovered unsaved work".to_owned(),
            0 => format!("Opened {}", jobs::display_name(path)),
            skipped => format!("Opened with {skipped} invalid retopology faces skipped"),
        });

        let Some(reference) = project.settings.reference else {
            // No sculpt recorded: keep whatever is loaded.
            self.saved_reference = self.reference_path.clone();
            if let Some(camera) = &project.settings.camera {
                self.apply_camera(camera);
            }
            return;
        };
        let found = reference
            .resolve(path)
            .or_else(|| self.locate_reference(&reference));
        // The project points at the recorded sculpt; a relocated one is a change to save.
        self.saved_reference = if recovered {
            None
        } else {
            reference.resolve(path)
        };
        match found {
            Some(mesh) if Some(&mesh) == self.reference_path.as_ref() => {
                if let Some(camera) = &project.settings.camera {
                    self.apply_camera(camera);
                }
            }
            Some(mesh) => {
                self.pending_camera = project.settings.camera;
                self.start_job(Job::Open(mesh));
            }
            None => {
                self.ui.error = Some(format!(
                    "The sculpt {} was not found; the retopology is shown without it.",
                    reference.file_name().unwrap_or_default()
                ));
                if let Some(camera) = &project.settings.camera {
                    self.apply_camera(camera);
                }
            }
        }
    }

    /// Asks the user to find a reference mesh that moved.
    fn locate_reference(&self, reference: &ReferencePath) -> Option<PathBuf> {
        let name = reference.file_name().unwrap_or_else(|| "the sculpt".into());
        let answer = rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Sculpt not found")
            .set_description(format!(
                "The project's sculpt, {name}, was not found. Locate it?"
            ))
            .set_buttons(rfd::MessageButtons::YesNo)
            .set_parent(self.window.as_ref())
            .show();
        if answer != rfd::MessageDialogResult::Yes {
            return None;
        }
        rfd::FileDialog::new()
            .set_title(format!("Locate {name}"))
            .add_filter("Meshes", &["obj"])
            .set_parent(self.window.as_ref())
            .pick_file()
    }

    pub(crate) fn new_project(&mut self) {
        if self.has_unsaved_changes()
            && !self.confirm_discard("Starting a new project replaces the current work.")
        {
            return;
        }
        self.retopo.replace_mesh(HalfEdgeMesh::new());
        self.tools.reset();
        self.project_path = None;
        self.saved_version = self.retopo.mesh().version();
        self.saved_reference = self.reference_path.clone();
        self.tools.status = Some("New project".into());
    }

    pub(crate) fn import_retopo(&mut self) {
        if !self.retopo.mesh().is_empty()
            && self.has_unsaved_changes()
            && !self.confirm_discard("Importing replaces the current retopology.")
        {
            return;
        }
        let picked = rfd::FileDialog::new()
            .set_title("Import retopology")
            .add_filter("OBJ", &["obj"])
            .set_parent(self.window.as_ref())
            .pick_file();
        let Some(path) = picked else { return };
        match polyflow_io::retopo::load(&path) {
            Ok(imported) => {
                let faces = imported.mesh.face_count();
                self.retopo.replace_mesh(imported.mesh);
                self.tools.reset();
                // Imported geometry is not saved in the project yet.
                self.saved_version = u64::MAX;
                self.tools.status = Some(match imported.skipped_faces {
                    0 => format!("Imported {faces} faces"),
                    skipped => format!("Imported {faces} faces; skipped {skipped} invalid ones"),
                });
            }
            Err(e) => self.ui.error = Some(format!("Could not import {}: {e}", path.display())),
        }
    }

    pub(crate) fn export_retopo(&mut self) {
        let default_name = self
            .project_path
            .as_deref()
            .and_then(|p| p.file_stem())
            .map_or("retopo".to_owned(), |s| s.to_string_lossy().into_owned());
        let picked = rfd::FileDialog::new()
            .set_title("Export retopology")
            .add_filter("OBJ", &["obj"])
            .set_file_name(format!("{default_name}.obj"))
            .set_parent(self.window.as_ref())
            .save_file();
        let Some(path) = picked else { return };
        match polyflow_io::retopo::save(&path, self.retopo.mesh()) {
            Ok(()) => self.tools.status = Some(format!("Exported {}", jobs::display_name(&path))),
            Err(e) => self.ui.error = Some(format!("Could not export {}: {e}", path.display())),
        }
    }

    /// Asks before throwing away unsaved work. True to go ahead.
    pub(crate) fn confirm_discard(&mut self, what: &str) -> bool {
        let answer = rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Unsaved changes")
            .set_description(format!("{what} Save the project first?"))
            .set_buttons(rfd::MessageButtons::YesNoCancel)
            .set_parent(self.window.as_ref())
            .show();
        match answer {
            rfd::MessageDialogResult::Yes => self.save_project(false),
            rfd::MessageDialogResult::No => true,
            _ => false,
        }
    }

    /// True if the app may close (nothing unsaved, or the user decided).
    pub(crate) fn confirm_close(&mut self) -> bool {
        !self.has_unsaved_changes() || self.confirm_discard("The project has unsaved changes.")
    }

    pub(crate) fn update_title(&mut self) {
        let mut title = String::new();
        if let Some(path) = &self.project_path {
            title += &jobs::display_name(path);
        } else if !self.retopo.mesh().is_empty() {
            title += "Untitled";
        }
        if self.has_unsaved_changes() {
            title += "*";
        }
        if !title.is_empty() {
            title += " · ";
        }
        if self.mesh_info.triangles > 0 {
            title += &self.mesh_info.name;
            title += " — ";
        }
        title += "Polyflow";
        if title != self.title {
            self.window.set_title(&title);
            self.title = title;
        }
    }

    // ---- Autosave and recovery -------------------------------------------------

    /// Writes unsaved work to the recovery folder when due. Returns when to
    /// check again, if there is anything unsaved.
    pub(crate) fn autosave_if_due(&mut self) -> Option<Instant> {
        if !self.autosave.enabled || !self.has_unsaved_changes() || self.retopo.mesh().is_empty() {
            return None;
        }
        let now = Instant::now();
        if now < self.autosave.next {
            return Some(self.autosave.next);
        }
        self.autosave.next = now + autosave_interval();
        let state = (self.retopo.mesh().version(), self.reference_path.clone());
        if self.autosave.saved_state.as_ref() == Some(&state) {
            return Some(self.autosave.next);
        }
        let path = autosave_path()?;
        let result = std::fs::create_dir_all(path.parent().expect("autosave path has a folder"))
            .map_err(project::ProjectError::from)
            .and_then(|()| project::save(&path, &self.project_settings(&path), self.retopo.mesh()));
        match result {
            Ok(()) => {
                log::info!("Autosaved to {}", path.display());
                self.autosave.saved_state = Some(state);
            }
            Err(e) => log::warn!("Autosave failed: {e}"),
        }
        Some(self.autosave.next)
    }

    /// Deletes this session's autosave (after saving, or on a clean exit).
    pub(crate) fn remove_autosave(&mut self) {
        self.autosave.saved_state = None;
        self.autosave.next = Instant::now() + autosave_interval();
        if let Some(path) = autosave_path() {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Offers to restore work autosaved by a session that did not close
    /// cleanly. Returns true if it was restored.
    pub(crate) fn offer_recovery(&mut self) -> bool {
        let stale = stale_autosaves();
        let Some((newest, modified)) = stale.first().cloned() else {
            return false;
        };
        let answer = rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Info)
            .set_title("Recover unsaved work")
            .set_description(format!(
                "Polyflow did not close properly last time. Recover the work autosaved {}?",
                describe_age(modified)
            ))
            .set_buttons(rfd::MessageButtons::YesNo)
            .set_parent(self.window.as_ref())
            .show();
        let recovered = answer == rfd::MessageDialogResult::Yes;
        if recovered {
            self.open_project(&newest, true);
        }
        // Either way the old files are no longer needed: recovered work is
        // autosaved again under this session.
        for (path, _) in stale {
            let _ = std::fs::remove_file(path);
        }
        recovered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ages_read_naturally() {
        let now = SystemTime::now();
        assert_eq!(describe_age(now), "less than a minute ago");
        assert_eq!(
            describe_age(now - Duration::from_secs(5 * 60)),
            "5 minutes ago"
        );
        assert_eq!(
            describe_age(now - Duration::from_secs(3 * 3600)),
            "3 hours ago"
        );
    }
}
