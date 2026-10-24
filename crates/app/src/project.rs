//! Project files, unsaved-change tracking, autosave and crash recovery.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use polyflow_bake::Cage;
use polyflow_core::camera::OrbitCamera;
use polyflow_core::glam::Vec3;
use polyflow_core::halfedge::HalfEdgeMesh;
use polyflow_io::project::{
    self, CageState, CameraState, ProjectSettings, ReferencePath, ToolState,
};
use polyflow_io::retopo::ImportedRetopo;

use crate::AppState;
use crate::jobs::{self, Job};
use crate::retopo::symmetry::Axis;
use crate::retopo::tools::ToolSettings;

/// How often unsaved work is written to the recovery folder.
pub const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(120);
const AUTOSAVE_PREFIX: &str = "autosave-";
/// Retopologies above this many faces are probably sculpts imported by
/// mistake; importing asks whether to load them as the reference instead.
const SCULPT_FACES: usize = 500_000;

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
            uv_checker: self.retopo.uv_checker,
            cage: CageState {
                push: self.retopo.cage_push,
                show: self.retopo.show_cage,
            },
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
        self.retopo.uv_checker = settings.uv_checker;
        self.retopo.cage_push = settings.cage.push;
        self.retopo.show_cage = settings.cage.show;
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
    //
    // Dialogs answer asynchronously (see `modal`), so commands that ask
    // something continue in a closure.

    /// Saves the project, asking for a file if there is none yet or `ask` is
    /// set, then runs `then` if it was saved.
    pub(crate) fn save_project(&mut self, ask: bool, then: impl FnOnce(&mut Self) + 'static) {
        if let (Some(path), false) = (&self.project_path, ask) {
            if self.save_project_to(path.clone()) {
                then(self);
            }
            return;
        }
        let default_name = self
            .project_path
            .as_deref()
            .or(self.reference_path.as_deref())
            .and_then(|p| p.file_stem())
            .map_or("untitled".to_owned(), |s| s.to_string_lossy().into_owned());
        self.ask(
            move |parent| {
                rfd::FileDialog::new()
                    .set_title("Save project")
                    .add_filter("Polyflow project", &[project::EXTENSION])
                    .set_file_name(format!("{default_name}.{}", project::EXTENSION))
                    .set_parent(parent)
                    .save_file()
            },
            move |state, picked| {
                if let Some(path) = picked
                    && state.save_project_to(path.with_extension(project::EXTENSION))
                {
                    then(state);
                }
            },
        );
    }

    /// Writes the project to `path`. Returns whether it was saved.
    fn save_project_to(&mut self, path: PathBuf) -> bool {
        let settings = self.project_settings(&path);
        let cage = self.imported_cage_mesh();
        match project::save(&path, &settings, self.retopo.mesh(), cage.as_ref()) {
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
        self.confirm_discard("Opening a project replaces the current work.", |state| {
            state.ask(
                |parent| {
                    rfd::FileDialog::new()
                        .set_title("Open project")
                        .add_filter("Polyflow project", &[project::EXTENSION])
                        .set_parent(parent)
                        .pick_file()
                },
                |state, picked| {
                    if let Some(path) = picked {
                        state.open_project(&path, false);
                    }
                },
            );
        });
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
        if let Some(cage) = &project.cage {
            match Cage::from_mesh(self.retopo.mesh(), cage) {
                Ok(cage) => self.retopo.set_imported_cage(Some(cage)),
                Err(e) => {
                    self.ui.error = Some(format!(
                        "The project's bake cage was not loaded ({e}); the automatic cage is used."
                    ))
                }
            }
        }
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

        let camera = project.settings.camera;
        let Some(reference) = project.settings.reference else {
            // No sculpt recorded: keep whatever is loaded.
            self.saved_reference = self.reference_path.clone();
            if let Some(camera) = &camera {
                self.apply_camera(camera);
            }
            return;
        };
        let recorded = reference.resolve(path);
        // The project points at the recorded sculpt; a relocated one is a change to save.
        self.saved_reference = if recovered { None } else { recorded.clone() };
        let name = reference.file_name().unwrap_or_else(|| "the sculpt".into());
        match recorded {
            Some(mesh) => self.use_project_reference(Some(mesh), &name, camera),
            None => self.locate_reference(name, camera),
        }
    }

    /// Asks the user to find a reference mesh that moved, then loads it.
    fn locate_reference(&mut self, name: String, camera: Option<CameraState>) {
        let description = format!("The project's sculpt, {name}, was not found. Locate it?");
        self.ask(
            move |parent| {
                rfd::MessageDialog::new()
                    .set_level(rfd::MessageLevel::Warning)
                    .set_title("Sculpt not found")
                    .set_description(description)
                    .set_buttons(rfd::MessageButtons::YesNo)
                    .set_parent(parent)
                    .show()
            },
            move |state, answer| {
                if answer != rfd::MessageDialogResult::Yes {
                    state.use_project_reference(None, &name, camera);
                    return;
                }
                let title = format!("Locate {name}");
                state.ask(
                    move |parent| {
                        rfd::FileDialog::new()
                            .set_title(title)
                            .add_filter("Meshes", &["obj"])
                            .set_parent(parent)
                            .pick_file()
                    },
                    move |state, picked| state.use_project_reference(picked, &name, camera),
                );
            },
        );
    }

    /// Shows the project's sculpt (`None` if it could not be found) with the
    /// project's camera.
    fn use_project_reference(
        &mut self,
        mesh: Option<PathBuf>,
        name: &str,
        camera: Option<CameraState>,
    ) {
        match mesh {
            Some(mesh) if Some(&mesh) == self.reference_path.as_ref() => {
                if let Some(camera) = &camera {
                    self.apply_camera(camera);
                }
            }
            Some(mesh) => {
                self.pending_camera = camera;
                self.start_job(Job::Open(mesh));
            }
            None => {
                self.ui.error = Some(format!(
                    "The sculpt {name} was not found; the retopology is shown without it."
                ));
                if let Some(camera) = &camera {
                    self.apply_camera(camera);
                }
            }
        }
    }

    pub(crate) fn new_project(&mut self) {
        self.confirm_discard(
            "Starting a new project replaces the current work.",
            |state| {
                state.retopo.replace_mesh(HalfEdgeMesh::new());
                state.tools.reset();
                state.project_path = None;
                state.saved_version = state.retopo.mesh().version();
                state.saved_reference = state.reference_path.clone();
                state.tools.status = Some("New project".into());
            },
        );
    }

    pub(crate) fn import_retopo(&mut self) {
        let pick = |state: &mut Self| {
            state.ask(
                |parent| {
                    rfd::FileDialog::new()
                        .set_title("Import retopology")
                        .add_filter("OBJ", &["obj"])
                        .set_parent(parent)
                        .pick_file()
                },
                |state, picked| {
                    if let Some(path) = picked {
                        state.load_retopo(path);
                    }
                },
            );
        };
        // An empty retopology has nothing to lose.
        if self.retopo.mesh().is_empty() {
            pick(self);
        } else {
            self.confirm_discard("Importing replaces the current retopology.", pick);
        }
    }

    fn load_retopo(&mut self, path: PathBuf) {
        let imported = match polyflow_io::retopo::load(&path) {
            Ok(imported) => imported,
            Err(e) => {
                self.ui.error = Some(format!("Could not import {}: {e}", path.display()));
                return;
            }
        };
        let faces = imported.mesh.face_count();
        if faces <= SCULPT_FACES {
            self.use_imported_retopo(imported);
            return;
        }
        let description = format!(
            "{} has {faces} faces: it looks like a sculpt, not a retopology. \
             Load it as the reference mesh instead?",
            jobs::display_name(&path)
        );
        self.ask(
            move |parent| {
                rfd::MessageDialog::new()
                    .set_level(rfd::MessageLevel::Warning)
                    .set_title("Import retopology")
                    .set_description(description)
                    .set_buttons(rfd::MessageButtons::YesNoCancel)
                    .set_parent(parent)
                    .show()
            },
            move |state, answer| match answer {
                rfd::MessageDialogResult::Yes => state.start_job(Job::Open(path)),
                rfd::MessageDialogResult::No => state.use_imported_retopo(imported),
                _ => {}
            },
        );
    }

    fn use_imported_retopo(&mut self, imported: ImportedRetopo) {
        let faces = imported.mesh.face_count();
        let has_uvs = imported.mesh.has_uvs();
        self.retopo.replace_mesh(imported.mesh);
        // Show freshly imported UVs; keep the user's choice otherwise.
        self.retopo.uv_checker |= has_uvs;
        self.tools.reset();
        // Imported geometry is not saved in the project yet.
        self.saved_version = u64::MAX;
        let uvs = if has_uvs { " with UVs" } else { "" };
        self.tools.status = Some(match imported.skipped_faces {
            0 => format!("Imported {faces} faces{uvs}"),
            skipped => format!("Imported {faces} faces{uvs}; skipped {skipped} invalid ones"),
        });
    }

    pub(crate) fn export_retopo(&mut self) {
        let default_name = self
            .project_path
            .as_deref()
            .and_then(|p| p.file_stem())
            .map_or("retopo".to_owned(), |s| s.to_string_lossy().into_owned());
        self.ask(
            move |parent| {
                rfd::FileDialog::new()
                    .set_title("Export retopology")
                    .add_filter("OBJ", &["obj"])
                    .set_file_name(format!("{default_name}.obj"))
                    .set_parent(parent)
                    .save_file()
            },
            |state, picked| {
                let Some(path) = picked else { return };
                match polyflow_io::retopo::save(&path, state.retopo.mesh()) {
                    Ok(()) => {
                        state.tools.status = Some(format!("Exported {}", jobs::display_name(&path)))
                    }
                    Err(e) => {
                        state.ui.error = Some(format!("Could not export {}: {e}", path.display()))
                    }
                }
            },
        );
    }

    /// The imported cage as a mesh to save, if there is one that fits.
    fn imported_cage_mesh(&self) -> Option<HalfEdgeMesh> {
        let mesh = self.retopo.mesh();
        self.retopo
            .imported_cage()
            .filter(|c| c.fits(mesh))
            .map(|c| c.apply(mesh))
    }

    pub(crate) fn import_cage(&mut self) {
        self.ask(
            |parent| {
                rfd::FileDialog::new()
                    .set_title("Import bake cage")
                    .add_filter("OBJ", &["obj"])
                    .set_parent(parent)
                    .pick_file()
            },
            |state, picked| {
                let Some(path) = picked else { return };
                let cage = polyflow_io::retopo::load(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|c| {
                        Cage::from_mesh(state.retopo.mesh(), &c.mesh).map_err(|e| e.to_string())
                    });
                match cage {
                    Ok(cage) => {
                        state.retopo.set_imported_cage(Some(cage));
                        state.retopo.show_cage = true;
                        // The cage is saved with the project.
                        state.saved_version = u64::MAX;
                        state.tools.status =
                            Some(format!("Imported the cage {}", jobs::display_name(&path)));
                    }
                    Err(e) => {
                        state.ui.error = Some(format!(
                            "{} does not fit the retopology: {e}. A cage must be the retopology \
                             exported by Polyflow with vertices moved, none added, removed or \
                             reordered.",
                            jobs::display_name(&path)
                        ))
                    }
                }
            },
        );
    }

    /// Exports the cage in use (imported or automatic) as OBJ, to edit it in
    /// another program and import it back.
    pub(crate) fn export_cage(&mut self) {
        if self.retopo.mesh().is_empty() {
            self.ui.error = Some("There is no retopology to export a cage of.".into());
            return;
        }
        let default_name = self
            .project_path
            .as_deref()
            .and_then(|p| p.file_stem())
            .map_or("retopo".to_owned(), |s| s.to_string_lossy().into_owned());
        self.ask(
            move |parent| {
                rfd::FileDialog::new()
                    .set_title("Export bake cage")
                    .add_filter("OBJ", &["obj"])
                    .set_file_name(format!("{default_name}_cage.obj"))
                    .set_parent(parent)
                    .save_file()
            },
            |state, picked| {
                let Some(path) = picked else { return };
                let cage = state.retopo.cage().apply(state.retopo.mesh());
                match polyflow_io::retopo::save(&path, &cage) {
                    Ok(()) => {
                        state.tools.status =
                            Some(format!("Exported the cage {}", jobs::display_name(&path)))
                    }
                    Err(e) => {
                        state.ui.error = Some(format!("Could not export {}: {e}", path.display()))
                    }
                }
            },
        );
    }

    /// Runs `then` once unsaved work may be thrown away: right away if there
    /// is none, otherwise after asking (and saving, if the user wants to).
    pub(crate) fn confirm_discard(&mut self, what: &str, then: impl FnOnce(&mut Self) + 'static) {
        if !self.has_unsaved_changes() {
            then(self);
            return;
        }
        let description = format!("{what} Save the project first?");
        self.ask(
            move |parent| {
                rfd::MessageDialog::new()
                    .set_level(rfd::MessageLevel::Warning)
                    .set_title("Unsaved changes")
                    .set_description(description)
                    .set_buttons(rfd::MessageButtons::YesNoCancel)
                    .set_parent(parent)
                    .show()
            },
            move |state, answer| match answer {
                rfd::MessageDialogResult::Yes => state.save_project(false, then),
                rfd::MessageDialogResult::No => then(state),
                _ => {}
            },
        );
    }

    /// Closes the app, asking first if there is unsaved work.
    pub(crate) fn request_close(&mut self) {
        self.confirm_discard("The project has unsaved changes.", |state| {
            state.exit_requested = true;
        });
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
            .and_then(|()| {
                let cage = self.imported_cage_mesh();
                project::save(
                    &path,
                    &self.project_settings(&path),
                    self.retopo.mesh(),
                    cage.as_ref(),
                )
            });
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
    /// Offers to recover work autosaved by a session that did not close
    /// properly; runs `otherwise` if there is none or the user declines.
    pub(crate) fn offer_recovery(&mut self, otherwise: impl FnOnce(&mut Self) + 'static) {
        let stale = stale_autosaves();
        let Some((newest, modified)) = stale.first().cloned() else {
            otherwise(self);
            return;
        };
        let description = format!(
            "Polyflow did not close properly last time. Recover the work autosaved {}?",
            describe_age(modified)
        );
        self.ask(
            move |parent| {
                rfd::MessageDialog::new()
                    .set_level(rfd::MessageLevel::Info)
                    .set_title("Recover unsaved work")
                    .set_description(description)
                    .set_buttons(rfd::MessageButtons::YesNo)
                    .set_parent(parent)
                    .show()
            },
            move |state, answer| {
                if answer == rfd::MessageDialogResult::Yes {
                    state.open_project(&newest, true);
                } else {
                    otherwise(state);
                }
                // Either way the old files are no longer needed: recovered
                // work is autosaved again under this session.
                for (path, _) in stale {
                    let _ = std::fs::remove_file(path);
                }
            },
        );
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
