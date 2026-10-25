//! Command-line options.

use std::path::PathBuf;
use std::time::Duration;

pub const USAGE: &str = "\
Usage: polyflow [OPTIONS] [MESH]

Arguments:
  [MESH]               Reference mesh to open (.obj, .ply, .gltf, .glb, .fbx)

Options:
  --torus <RINGS>      Start with a test torus of 2*RINGS*(RINGS/2) triangles
  --no-vsync           Disable vsync
  --continuous         Redraw every frame instead of only on changes
  --zoom <FACTOR>      Move the camera FACTOR times closer than the framed view
  --bench <SECONDS>    Spin the camera for SECONDS (implies --no-vsync and
                       --continuous), print frame time statistics and exit
  -h, --help           Print this help";

#[derive(Debug, Default, PartialEq)]
pub struct Options {
    pub mesh: Option<PathBuf>,
    pub torus_rings: Option<u32>,
    pub vsync: bool,
    pub continuous: bool,
    pub bench: Option<Duration>,
    pub zoom: Option<f32>,
}

#[derive(Debug, PartialEq)]
pub enum ParseResult {
    Run(Options),
    Help,
    Error(String),
}

pub fn parse(args: impl IntoIterator<Item = std::ffi::OsString>) -> ParseResult {
    let mut options = Options {
        vsync: true,
        ..Default::default()
    };
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let Some(flag) = arg.to_str() else {
            options.mesh = Some(arg.into());
            continue;
        };
        let mut value = |name: &str| {
            args.next()
                .and_then(|v| v.into_string().ok())
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match flag {
            "-h" | "--help" => return ParseResult::Help,
            "--no-vsync" => options.vsync = false,
            "--continuous" => options.continuous = true,
            "--torus" => match value(flag).map(|v| v.parse::<u32>()) {
                Ok(Ok(rings)) if rings >= 3 => options.torus_rings = Some(rings),
                Ok(_) => {
                    return ParseResult::Error("--torus needs an integer of at least 3".into());
                }
                Err(e) => return ParseResult::Error(e),
            },
            "--zoom" => match value(flag).map(|v| v.parse::<f32>()) {
                Ok(Ok(factor)) if factor > 0.0 && factor.is_finite() => options.zoom = Some(factor),
                Ok(_) => return ParseResult::Error("--zoom needs a positive number".into()),
                Err(e) => return ParseResult::Error(e),
            },
            "--bench" => match value(flag).map(|v| v.parse::<f32>()) {
                Ok(Ok(secs)) if secs > 0.0 && secs.is_finite() => {
                    options.bench = Some(Duration::from_secs_f32(secs));
                    options.vsync = false;
                    options.continuous = true;
                }
                Ok(_) => {
                    return ParseResult::Error("--bench needs a positive number of seconds".into());
                }
                Err(e) => return ParseResult::Error(e),
            },
            _ if flag.starts_with('-') => {
                return ParseResult::Error(format!("unknown option {flag}"));
            }
            _ if options.mesh.is_some() => {
                return ParseResult::Error("only one mesh can be opened".into());
            }
            _ => options.mesh = Some(flag.into()),
        }
    }
    ParseResult::Run(options)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<std::ffi::OsString> {
        list.iter().map(Into::into).collect()
    }

    #[test]
    fn parses_options() {
        assert_eq!(
            parse(args(&["sculpt.obj", "--no-vsync"])),
            ParseResult::Run(Options {
                mesh: Some("sculpt.obj".into()),
                vsync: false,
                ..Default::default()
            })
        );
        let ParseResult::Run(o) = parse(args(&["--torus", "100", "--bench", "5"])) else {
            panic!()
        };
        assert_eq!(o.torus_rings, Some(100));
        assert_eq!(o.bench, Some(Duration::from_secs(5)));
        assert!(!o.vsync && o.continuous);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(matches!(parse(args(&["--torus"])), ParseResult::Error(_)));
        assert!(matches!(
            parse(args(&["--torus", "2"])),
            ParseResult::Error(_)
        ));
        assert!(matches!(
            parse(args(&["--bench", "-1"])),
            ParseResult::Error(_)
        ));
        assert!(matches!(
            parse(args(&["--frobnicate"])),
            ParseResult::Error(_)
        ));
        assert!(matches!(
            parse(args(&["a.obj", "b.obj"])),
            ParseResult::Error(_)
        ));
        assert_eq!(parse(args(&["-h"])), ParseResult::Help);
    }
}
