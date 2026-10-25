//! Polyflow project files (`.pflow`).
//!
//! A project is a zip archive with two entries:
//! - `project.json`: the format version, where the reference mesh is, and the
//!   view and tool settings;
//! - `retopo.obj`: the retopology mesh as polygon OBJ, with its UVs;
//! - `cage.obj` (optional): an imported bake cage, the retopology with moved
//!   vertices.
//!
//! The reference mesh is not embedded (sculpts can be gigabytes). Its path is
//! stored both relative to the project and absolute, so a project keeps
//! working when its folder is moved together with the sculpt.
//!
//! Compatibility: unknown JSON fields are ignored and missing ones take their
//! defaults, so newer minor additions stay readable. Breaking changes bump
//! [`FORMAT_VERSION`]; files from a newer version are refused with a clear error.

use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use polyflow_core::halfedge::HalfEdgeMesh;
use serde::{Deserialize, Serialize};

use crate::obj::ObjError;
use crate::retopo;

pub const EXTENSION: &str = "pflow";
pub const FORMAT_VERSION: u32 = 1;
const FORMAT_NAME: &str = "polyflow-project";
const MANIFEST: &str = "project.json";
const RETOPO: &str = "retopo.obj";
const CAGE: &str = "cage.obj";
/// Upper bounds on entry sizes, so a malicious archive cannot exhaust memory.
const MAX_MANIFEST_BYTES: u64 = 16 << 20;
const MAX_RETOPO_BYTES: u64 = 2 << 30;

#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("not a Polyflow project ({0})")]
    Archive(#[from] zip::result::ZipError),
    #[error("invalid project data: {0}")]
    Manifest(#[from] serde_json::Error),
    #[error("invalid retopology data: {0}")]
    Retopo(#[from] ObjError),
    #[error("not a Polyflow project")]
    NotAProject,
    #[error(
        "the project was made by a newer Polyflow (format {0}, this version reads up to {FORMAT_VERSION})"
    )]
    TooNew(u32),
}

/// Where the reference mesh is. Either path may be missing.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ReferencePath {
    /// Relative to the project file's folder, with `/` separators.
    pub relative: Option<String>,
    pub absolute: Option<String>,
}

impl ReferencePath {
    /// Describes `mesh` for a project saved at `project`.
    pub fn new(mesh: &Path, project: &Path) -> Self {
        let absolute = std::path::absolute(mesh).unwrap_or_else(|_| mesh.to_path_buf());
        let project_dir = project
            .parent()
            .map(|d| std::path::absolute(d).unwrap_or_else(|_| d.to_path_buf()));
        let relative = project_dir
            .and_then(|dir| relative_path(&absolute, &dir))
            .map(|p| {
                p.components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/")
            });
        Self {
            relative,
            absolute: Some(absolute.to_string_lossy().into_owned()),
        }
    }

    /// The first of the relative and absolute paths that exists.
    pub fn resolve(&self, project: &Path) -> Option<PathBuf> {
        let from_relative = self
            .relative
            .as_ref()
            .and_then(|r| project.parent().map(|dir| dir.join(r)));
        let from_absolute = self.absolute.as_ref().map(PathBuf::from);
        from_relative
            .into_iter()
            .chain(from_absolute)
            .find(|p| p.is_file())
    }

    /// File name to mention when the mesh cannot be found.
    pub fn file_name(&self) -> Option<String> {
        let path = self.absolute.as_ref().or(self.relative.as_ref())?;
        Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraState {
    pub target: [f32; 3],
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolState {
    pub strip_width: f32,
    pub brush_radius: f32,
    pub brush_strength: f32,
    pub relax_borders: bool,
}

impl Default for ToolState {
    fn default() -> Self {
        Self {
            strip_width: 40.0,
            brush_radius: 70.0,
            brush_strength: 0.35,
            relax_borders: false,
        }
    }
}

/// Everything in a project besides the retopology mesh.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProjectSettings {
    pub reference: Option<ReferencePath>,
    pub camera: Option<CameraState>,
    pub symmetry: Option<Axis>,
    pub x_ray: bool,
    pub surface_offset: f32,
    /// Show the retopology's UVs as a checker pattern.
    pub uv_checker: bool,
    pub tools: ToolState,
    pub cage: CageState,
    pub bake: BakeState,
}

/// Bake settings. Distances are in scene units; `None` derives them from
/// the sculpt's size.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BakeState {
    /// Map width and height.
    pub size: u32,
    /// Samples per texel along each axis.
    pub samples: u32,
    pub tangent_normal: bool,
    pub object_normal: bool,
    pub ambient_occlusion: bool,
    /// Green down in tangent-space normal maps.
    pub directx: bool,
    pub ao_samples: u32,
    pub ao_distance: Option<f32>,
    /// Start rays from the cage instead of `frontal` in front.
    pub use_cage: bool,
    pub frontal: Option<f32>,
    pub rear: Option<f32>,
    /// Texels of edge padding; ignored if `pad_full`.
    pub padding: u32,
    pub pad_full: bool,
    /// `polyflow_io::image::MapFormat` name.
    pub format: String,
    /// Output files are this path plus `_normal.png` and so on; `None`
    /// names them after the project, next to it.
    pub output: Option<String>,
    /// Bake on the CPU even with a GPU.
    pub cpu: bool,
    /// Split normals at UV seams too, besides edges marked hard.
    pub hard_uv_seams: bool,
}

impl Default for BakeState {
    fn default() -> Self {
        Self {
            size: 2048,
            samples: 1,
            tangent_normal: true,
            object_normal: false,
            ambient_occlusion: true,
            directx: false,
            ao_samples: 32,
            ao_distance: None,
            use_cage: false,
            frontal: None,
            rear: None,
            padding: 16,
            pad_full: false,
            format: crate::image::MapFormat::Png8.name().into(),
            output: None,
            cpu: false,
            hard_uv_seams: false,
        }
    }
}

/// How the bake cage is set up and shown.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CageState {
    /// How far the automatic cage is pushed out, in scene units; `None` to
    /// derive it from the sculpt's size.
    pub push: Option<f32>,
    pub show: bool,
}

impl Default for ProjectSettings {
    fn default() -> Self {
        Self {
            reference: None,
            camera: None,
            symmetry: None,
            x_ray: false,
            surface_offset: 0.01,
            uv_checker: false,
            tools: ToolState::default(),
            cage: CageState::default(),
            bake: BakeState::default(),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    format: String,
    format_version: u32,
    #[serde(default)]
    app_version: String,
    #[serde(flatten)]
    settings: ProjectSettings,
}

/// A loaded project.
#[derive(Debug)]
pub struct Project {
    pub settings: ProjectSettings,
    pub retopo: HalfEdgeMesh,
    /// Retopology faces that could not be restored (see [`retopo::ImportedRetopo`]).
    pub skipped_faces: usize,
    /// The imported cage, if the project has one.
    pub cage: Option<HalfEdgeMesh>,
}

/// Saves a project atomically: written next to `path`, then renamed into place.
pub fn save(
    path: &Path,
    settings: &ProjectSettings,
    retopo_mesh: &HalfEdgeMesh,
    cage: Option<&HalfEdgeMesh>,
) -> Result<(), ProjectError> {
    let mut temp = path.as_os_str().to_owned();
    temp.push(".tmp");
    let temp = PathBuf::from(temp);
    let result =
        write(&temp, settings, retopo_mesh, cage).and_then(|()| Ok(std::fs::rename(&temp, path)?));
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

fn write(
    path: &Path,
    settings: &ProjectSettings,
    retopo_mesh: &HalfEdgeMesh,
    cage: Option<&HalfEdgeMesh>,
) -> Result<(), ProjectError> {
    let file = std::fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(std::io::BufWriter::new(file));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    let manifest = Manifest {
        format: FORMAT_NAME.into(),
        format_version: FORMAT_VERSION,
        app_version: env!("CARGO_PKG_VERSION").into(),
        settings: settings.clone(),
    };
    zip.start_file(MANIFEST, options)?;
    serde_json::to_writer_pretty(&mut zip, &manifest)?;
    zip.start_file(RETOPO, options)?;
    retopo::write_obj(retopo_mesh, &mut zip)?;
    if let Some(cage) = cage {
        zip.start_file(CAGE, options)?;
        retopo::write_obj(cage, &mut zip)?;
    }
    let mut out = zip.finish()?;
    out.flush()?;
    out.get_ref().sync_all()?;
    Ok(())
}

pub fn load(path: &Path) -> Result<Project, ProjectError> {
    let file = std::fs::File::open(path)?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))?;

    let manifest: Manifest = {
        let entry = zip
            .by_name(MANIFEST)
            .map_err(|_| ProjectError::NotAProject)?;
        let mut text = String::new();
        entry.take(MAX_MANIFEST_BYTES).read_to_string(&mut text)?;
        serde_json::from_str(&text)?
    };
    if manifest.format != FORMAT_NAME {
        return Err(ProjectError::NotAProject);
    }
    if manifest.format_version > FORMAT_VERSION {
        return Err(ProjectError::TooNew(manifest.format_version));
    }

    let mut read_mesh = |name: &str| -> Result<Option<retopo::ImportedRetopo>, ProjectError> {
        match zip.by_name(name) {
            Ok(entry) => {
                let mut bytes = Vec::new();
                entry.take(MAX_RETOPO_BYTES).read_to_end(&mut bytes)?;
                Ok(Some(retopo::parse_obj(&bytes)?))
            }
            Err(zip::result::ZipError::FileNotFound) => Ok(None),
            Err(e) => Err(e.into()),
        }
    };
    // Missing when the project was saved before any retopology existed.
    let imported = read_mesh(RETOPO)?.unwrap_or(retopo::ImportedRetopo {
        mesh: HalfEdgeMesh::new(),
        skipped_faces: 0,
    });
    let cage = read_mesh(CAGE)?.map(|c| c.mesh);
    Ok(Project {
        settings: manifest.settings,
        retopo: imported.mesh,
        skipped_faces: imported.skipped_faces,
        cage,
    })
}

/// `path` relative to `base` (both absolute), if they share a root.
fn relative_path(path: &Path, base: &Path) -> Option<PathBuf> {
    let path: Vec<Component> = path.components().collect();
    let base: Vec<Component> = base.components().collect();
    // Different drives (Windows) or roots: no relative path.
    if path.first() != base.first() {
        return None;
    }
    let common = path.iter().zip(&base).take_while(|(a, b)| a == b).count();
    let mut relative = PathBuf::new();
    for _ in common..base.len() {
        relative.push("..");
    }
    for c in &path[common..] {
        relative.push(c.as_os_str());
    }
    Some(relative)
}

#[cfg(test)]
mod tests {
    use polyflow_core::glam::{Vec2, Vec3};

    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("polyflow-project-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    const QUAD_UVS: [Vec2; 4] = [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y];

    fn quad() -> HalfEdgeMesh {
        let mut mesh = HalfEdgeMesh::new();
        let v =
            [Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y].map(|p| mesh.add_vertex(p));
        let f = mesh.add_face(&v).unwrap();
        mesh.set_face_uvs(f, &QUAD_UVS);
        mesh
    }

    #[test]
    fn round_trip() {
        let dir = temp_dir("roundtrip");
        let sculpt = dir.join("meshes").join("sculpt.obj");
        std::fs::create_dir_all(sculpt.parent().unwrap()).unwrap();
        std::fs::write(&sculpt, "v 0 0 0\n").unwrap();
        let project_path = dir.join("work").join("character.pflow");
        std::fs::create_dir_all(project_path.parent().unwrap()).unwrap();

        let mut cage = quad();
        let ids: Vec<_> = cage.vertices().collect();
        for v in ids {
            let p = cage.position(v);
            cage.set_position(v, p + Vec3::Z * 0.5);
        }
        let settings = ProjectSettings {
            reference: Some(ReferencePath::new(&sculpt, &project_path)),
            camera: Some(CameraState {
                target: [1.0, 2.0, 3.0],
                distance: 4.5,
                yaw: 0.3,
                pitch: -0.2,
                fov_y: 0.8,
            }),
            symmetry: Some(Axis::X),
            x_ray: true,
            surface_offset: 0.02,
            uv_checker: true,
            cage: CageState {
                push: Some(0.25),
                show: true,
            },
            bake: BakeState {
                size: 4096,
                use_cage: true,
                output: Some("out/hero".into()),
                ..BakeState::default()
            },
            tools: ToolState {
                strip_width: 55.0,
                ..ToolState::default()
            },
        };
        save(&project_path, &settings, &quad(), Some(&cage)).unwrap();
        assert!(!dir.join("work").join("character.pflow.tmp").exists());

        let project = load(&project_path).unwrap();
        assert_eq!(project.settings, settings);
        assert_eq!(project.retopo.face_count(), 1);
        let cage_back = project.cage.as_ref().unwrap();
        assert_eq!(cage_back.vertex_count(), 4);
        assert!(cage_back.vertices().all(|v| cage_back.position(v).z == 0.5));
        let face = project.retopo.faces().next().unwrap();
        assert_eq!(project.retopo.face_uvs(face).unwrap(), QUAD_UVS);
        assert_eq!(
            project
                .settings
                .reference
                .as_ref()
                .unwrap()
                .relative
                .as_deref(),
            Some("../meshes/sculpt.obj")
        );
        assert_eq!(
            project
                .settings
                .reference
                .unwrap()
                .resolve(&project_path)
                .unwrap()
                .canonicalize()
                .unwrap(),
            sculpt.canonicalize().unwrap()
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn moved_folders_resolve_through_the_relative_path() {
        let dir = temp_dir("moved");
        let original = dir.join("a");
        std::fs::create_dir_all(&original).unwrap();
        std::fs::write(original.join("sculpt.obj"), "v 0 0 0\n").unwrap();
        let reference = ReferencePath::new(&original.join("sculpt.obj"), &original.join("p.pflow"));
        assert_eq!(reference.relative.as_deref(), Some("sculpt.obj"));

        // Move the folder: the absolute path breaks, the relative one still works.
        let moved = dir.join("b");
        std::fs::rename(&original, &moved).unwrap();
        assert_eq!(
            reference.resolve(&moved.join("p.pflow")),
            Some(moved.join("sculpt.obj"))
        );
        assert_eq!(
            reference.resolve(&dir.join("elsewhere").join("p.pflow")),
            None
        );
        assert_eq!(reference.file_name().as_deref(), Some("sculpt.obj"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn missing_fields_take_defaults_and_newer_formats_are_refused() {
        let dir = temp_dir("compat");
        let path = dir.join("p.pflow");
        let write_manifest = |json: &str| {
            let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
            zip.start_file(MANIFEST, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(json.as_bytes()).unwrap();
            zip.finish().unwrap();
        };

        write_manifest(
            r#"{"format": "polyflow-project", "format_version": 1, "future_field": [1, 2]}"#,
        );
        let project = load(&path).unwrap();
        assert_eq!(project.settings, ProjectSettings::default());
        assert_eq!(project.retopo.vertex_count(), 0);

        write_manifest(r#"{"format": "polyflow-project", "format_version": 2}"#);
        assert!(matches!(load(&path), Err(ProjectError::TooNew(2))));

        write_manifest(r#"{"format": "something-else", "format_version": 1}"#);
        assert!(matches!(load(&path), Err(ProjectError::NotAProject)));

        std::fs::write(&path, b"definitely not a zip").unwrap();
        assert!(matches!(load(&path), Err(ProjectError::Archive(_))));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn relative_paths() {
        let p = |s: &str| PathBuf::from(s);
        if cfg!(windows) {
            assert_eq!(
                relative_path(&p(r"C:\a\b\c.obj"), &p(r"C:\a\d")),
                Some(p(r"..\b\c.obj"))
            );
            assert_eq!(relative_path(&p(r"D:\a.obj"), &p(r"C:\a")), None);
        } else {
            assert_eq!(
                relative_path(&p("/a/b/c.obj"), &p("/a/d")),
                Some(p("../b/c.obj"))
            );
            assert_eq!(relative_path(&p("/a/b.obj"), &p("/a")), Some(p("b.obj")));
        }
    }
}
