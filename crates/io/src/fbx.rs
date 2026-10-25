//! FBX import of reference meshes, through ufbx.
//!
//! ufbx reads binary and ASCII FBX of any version and is built for untrusted
//! input. Every mesh in the scene is read, placed by its node, in Y-up
//! right-handed axes like the rest of Polyflow; the file's units are kept.
//! Polygons are fan-triangulated, as with the other formats.

use polyflow_core::glam::Vec3;

use crate::{Geometry, LoadError};

/// Parses FBX data.
pub fn parse(bytes: &[u8]) -> Result<Geometry, LoadError> {
    let options = ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        // Skip what a sculpt does not need.
        ignore_animation: true,
        ignore_embedded: true,
        load_external_files: false,
        ..Default::default()
    };
    let scene = ufbx::load_memory(bytes, options).map_err(|e| LoadError::Fbx(describe(&e)))?;
    let mut geometry = Geometry::default();
    let mut corners: Vec<u32> = Vec::new();
    for node in &scene.nodes {
        let Some(mesh) = &node.mesh else {
            continue;
        };
        let base = geometry.positions.len();
        if base + mesh.vertices.len() > u32::MAX as usize {
            return Err(LoadError::Fbx("the mesh is too large".into()));
        }
        let to_world = &node.geometry_to_world;
        geometry.positions.extend(mesh.vertices.iter().map(|&p| {
            let p = ufbx::transform_position(to_world, p);
            Vec3::new(p.x as f32, p.y as f32, p.z as f32)
        }));
        // A mirroring transform turns faces inside out: reverse them.
        let flip = determinant(to_world) < 0.0;
        for face in &mesh.faces {
            let range = face.index_begin as usize..(face.index_begin + face.num_indices) as usize;
            let Some(indices) = mesh.vertex_indices.get(range) else {
                geometry.skipped_faces += 1;
                continue;
            };
            corners.clear();
            corners.extend(indices.iter().map(|&i| base as u32 + i));
            if flip {
                corners.reverse();
            }
            geometry.push_polygon(&corners);
        }
    }
    if geometry.indices.len() > u32::MAX as usize {
        return Err(LoadError::Fbx("the mesh is too large".into()));
    }
    Ok(geometry)
}

fn determinant(m: &ufbx::Matrix) -> f64 {
    m.m00 * (m.m11 * m.m22 - m.m12 * m.m21) - m.m01 * (m.m10 * m.m22 - m.m12 * m.m20)
        + m.m02 * (m.m10 * m.m21 - m.m11 * m.m20)
}

fn describe(error: &ufbx::Error) -> String {
    let info = error.info();
    if info.is_empty() {
        error.description.to_string()
    } else {
        format!("{} ({info})", error.description.as_ref() as &str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quad and a mirrored copy of it, as ASCII FBX.
    const QUADS: &str = r#"; FBX 7.4.0 project file
FBXHeaderExtension:  {
	FBXHeaderVersion: 1003
	FBXVersion: 7400
}
Objects:  {
	Geometry: 1000, "Geometry::Quad", "Mesh" {
		Vertices: *12 {
			a: 0,0,0,1,0,0,1,1,0,0,1,0
		}
		PolygonVertexIndex: *4 {
			a: 0,1,2,-4
		}
	}
	Model: 2000, "Model::Quad", "Mesh" {
		Properties70:  {
			P: "Lcl Translation", "Lcl Translation", "", "A",5,0,0
		}
	}
	Model: 3000, "Model::Mirror", "Mesh" {
		Properties70:  {
			P: "Lcl Scaling", "Lcl Scaling", "", "A",-1,1,1
		}
	}
}
Connections:  {
	C: "OO",1000,2000
	C: "OO",1000,3000
	C: "OO",2000,0
	C: "OO",3000,0
}
"#;

    #[test]
    fn reads_meshes_in_place() {
        let g = parse(QUADS.as_bytes()).unwrap();
        assert_eq!(g.positions.len(), 8);
        assert_eq!(g.indices.len(), 12);
        assert!(
            g.positions.contains(&Vec3::new(6.0, 1.0, 0.0)),
            "{:?}",
            g.positions
        );
        assert!(
            g.positions.contains(&Vec3::new(-1.0, 1.0, 0.0)),
            "{:?}",
            g.positions
        );
        // Both copies face +Z after the mirror is undone.
        let mesh = g.into_mesh();
        for t in mesh.indices.as_chunks::<3>().0 {
            let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
            assert!((b - a).cross(c - a).z > 0.0);
        }
    }

    #[test]
    fn malformed_files_are_errors() {
        assert!(parse(b"").is_err());
        assert!(parse(b"Kaydara FBX Binary  \0\x1a\0garbage").is_err());
        let bytes = QUADS.as_bytes();
        for len in (0..bytes.len()).step_by(7) {
            let _ = parse(&bytes[..len]);
        }
    }
}
