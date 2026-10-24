//! Writing baked maps as image files.

use std::io::Write;
use std::path::Path;

/// Writes `data` (`channels` floats per pixel in 0..1, row by row from the
/// top) as an 8-bit PNG. Values are stored linearly, as normal and occlusion
/// maps expect; 1, 3 and 4 channels map to gray, RGB and RGBA.
pub fn write_png(
    out: impl Write,
    width: u32,
    height: u32,
    channels: usize,
    data: &[f32],
) -> Result<(), png::EncodingError> {
    assert_eq!(data.len(), width as usize * height as usize * channels);
    let color = match channels {
        1 => png::ColorType::Grayscale,
        3 => png::ColorType::Rgb,
        4 => png::ColorType::Rgba,
        _ => panic!("unsupported channel count {channels}"),
    };
    let mut encoder = png::Encoder::new(out, width, height);
    encoder.set_color(color);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let bytes: Vec<u8> = data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    writer.write_image_data(&bytes)?;
    writer.finish()
}

/// [`write_png`] to a file.
pub fn save_png(
    path: &Path,
    width: u32,
    height: u32,
    channels: usize,
    data: &[f32],
) -> std::io::Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    write_png(file, width, height, channels, data).map_err(std::io::Error::other)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_round_trip() {
        let data = [0.0, 0.5, 1.0, 2.0, -1.0, 0.25];
        let mut bytes = Vec::new();
        write_png(&mut bytes, 2, 1, 3, &data).unwrap();
        let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut reader = decoder.read_info().unwrap();
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut pixels).unwrap();
        assert_eq!(
            (info.width, info.height, info.color_type),
            (2, 1, png::ColorType::Rgb)
        );
        assert_eq!(&pixels[..6], &[0, 128, 255, 255, 0, 64]);
    }
}
