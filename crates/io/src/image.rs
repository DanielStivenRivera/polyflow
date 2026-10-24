//! Writing baked maps as image files.
//!
//! Maps are linear data (normals, occlusion), so no color space conversion
//! happens: PNGs store the values as they are, EXRs as linear floats.

use std::io::Write;
use std::path::Path;

use exr::prelude::{
    AnyChannel, AnyChannels, Encoding, FlatSamples, Image, SmallVec, WritableImage, f16,
};

/// File format and precision of a saved map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MapFormat {
    /// 8-bit PNG: small and universal; normal maps may band.
    #[default]
    Png8,
    /// 16-bit PNG: no banding, still a plain image.
    Png16,
    /// OpenEXR, 16-bit float.
    ExrHalf,
    /// OpenEXR, 32-bit float: the baker's full precision.
    ExrFloat,
}

impl MapFormat {
    pub const ALL: [MapFormat; 4] = [Self::Png8, Self::Png16, Self::ExrHalf, Self::ExrFloat];

    pub fn extension(self) -> &'static str {
        match self {
            Self::Png8 | Self::Png16 => "png",
            Self::ExrHalf | Self::ExrFloat => "exr",
        }
    }

    /// Short name, as on the command line.
    pub fn name(self) -> &'static str {
        match self {
            Self::Png8 => "png8",
            Self::Png16 => "png16",
            Self::ExrHalf => "exr16",
            Self::ExrFloat => "exr32",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.name() == name)
    }
}

/// Saves a map (`channels` floats per pixel, row by row from the top) as
/// `format`. PNG values are clamped to 0..1; 1, 3 and 4 channels map to
/// gray, RGB and RGBA (EXR: Y, RGB, RGBA).
pub fn save_map(
    path: &Path,
    format: MapFormat,
    width: u32,
    height: u32,
    channels: usize,
    data: &[f32],
) -> std::io::Result<()> {
    assert_eq!(data.len(), width as usize * height as usize * channels);
    match format {
        MapFormat::Png8 | MapFormat::Png16 => {
            let file = std::io::BufWriter::new(std::fs::File::create(path)?);
            write_png(
                file,
                format == MapFormat::Png16,
                width,
                height,
                channels,
                data,
            )
            .map_err(std::io::Error::other)
        }
        MapFormat::ExrHalf | MapFormat::ExrFloat => {
            exr_image(format == MapFormat::ExrHalf, width, height, channels, data)
                .write()
                .to_file(path)
                .map_err(std::io::Error::other)
        }
    }
}

/// Writes a PNG of 8 or 16 bits per channel.
pub fn write_png(
    out: impl Write,
    sixteen_bit: bool,
    width: u32,
    height: u32,
    channels: usize,
    data: &[f32],
) -> Result<(), png::EncodingError> {
    let color = match channels {
        1 => png::ColorType::Grayscale,
        3 => png::ColorType::Rgb,
        4 => png::ColorType::Rgba,
        _ => panic!("unsupported channel count {channels}"),
    };
    let mut encoder = png::Encoder::new(out, width, height);
    encoder.set_color(color);
    let bytes: Vec<u8> = if sixteen_bit {
        encoder.set_depth(png::BitDepth::Sixteen);
        data.iter()
            .flat_map(|&v| ((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_be_bytes())
            .collect()
    } else {
        encoder.set_depth(png::BitDepth::Eight);
        data.iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    };
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&bytes)?;
    writer.finish()
}

fn exr_image(
    half: bool,
    width: u32,
    height: u32,
    channels: usize,
    data: &[f32],
) -> Image<exr::prelude::Layer<AnyChannels<FlatSamples>>> {
    let names: &[&str] = match channels {
        1 => &["Y"],
        3 => &["R", "G", "B"],
        4 => &["R", "G", "B", "A"],
        _ => panic!("unsupported channel count {channels}"),
    };
    let list: SmallVec<[AnyChannel<FlatSamples>; 4]> = names
        .iter()
        .enumerate()
        .map(|(c, name)| {
            let values = data.iter().skip(c).step_by(channels).copied();
            let samples = if half {
                FlatSamples::F16(values.map(f16::from_f32).collect())
            } else {
                FlatSamples::F32(values.collect())
            };
            AnyChannel::new(*name, samples)
        })
        .collect();
    Image::from_encoded_channels(
        (width as usize, height as usize),
        Encoding::SMALL_LOSSLESS,
        AnyChannels::sort(list),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_png(bytes: Vec<u8>) -> (png::OutputInfo, Vec<u8>) {
        let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut reader = decoder.read_info().unwrap();
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut pixels).unwrap();
        (info, pixels)
    }

    #[test]
    fn png_round_trip() {
        let data = [0.0, 0.5, 1.0, 2.0, -1.0, 0.25];
        let mut bytes = Vec::new();
        write_png(&mut bytes, false, 2, 1, 3, &data).unwrap();
        let (info, pixels) = decode_png(bytes);
        assert_eq!(
            (info.width, info.height, info.color_type),
            (2, 1, png::ColorType::Rgb)
        );
        assert_eq!(&pixels[..6], &[0, 128, 255, 255, 0, 64]);

        let mut bytes = Vec::new();
        write_png(&mut bytes, true, 1, 2, 1, &[0.5, 1.0]).unwrap();
        let (info, pixels) = decode_png(bytes);
        assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
        assert_eq!(pixels, [0x80, 0x00, 0xff, 0xff]);
    }

    #[test]
    fn exr_round_trip() {
        let dir = std::env::temp_dir().join(format!("polyflow-exr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Three columns, two rows; the top row first.
        let data: Vec<f32> = (0..18).map(|i| i as f32 * 0.1 - 0.3).collect();
        for format in [MapFormat::ExrHalf, MapFormat::ExrFloat] {
            let path = dir.join(format!("{}.exr", format.name()));
            save_map(&path, format, 3, 2, 3, &data).unwrap();
            let image = exr::prelude::read_first_flat_layer_from_file(&path).unwrap();
            let layer = image.layer_data;
            assert_eq!(layer.size, exr::prelude::Vec2(3, 2));
            let names: Vec<String> = layer
                .channel_data
                .list
                .iter()
                .map(|c| c.name.to_string())
                .collect();
            assert_eq!(names, ["B", "G", "R"]);
            for (c, channel) in layer.channel_data.list.iter().enumerate() {
                let source = 2 - c; // B, G, R
                let values: Vec<f32> = channel.sample_data.values_as_f32().collect();
                for (i, v) in values.iter().enumerate() {
                    let expected = data[i * 3 + source];
                    let tolerance = if format == MapFormat::ExrHalf {
                        1e-3
                    } else {
                        0.0
                    };
                    assert!(
                        (v - expected).abs() <= tolerance,
                        "{format:?} {i}: {v} vs {expected}"
                    );
                }
            }
        }
        let gray = dir.join("gray.exr");
        save_map(&gray, MapFormat::ExrFloat, 2, 1, 1, &[0.25, 0.75]).unwrap();
        let image = exr::prelude::read_first_flat_layer_from_file(&gray).unwrap();
        assert_eq!(image.layer_data.channel_data.list[0].name.to_string(), "Y");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn format_names() {
        for f in MapFormat::ALL {
            assert_eq!(MapFormat::from_name(f.name()), Some(f));
        }
        assert_eq!(MapFormat::from_name("tga"), None);
    }
}
