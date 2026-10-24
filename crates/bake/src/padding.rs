//! Edge padding: extending UV islands into the empty texels around them.
//!
//! Texture filtering and mipmaps blend texels across island borders, so the
//! background must continue the islands' values or seams show up as lines.
//! The islands grow one ring of texels at a time (diagonal steps count as
//! one), up to `padding` rings: each new texel takes the average of its
//! neighbors from earlier rings. Averaging, rather than copying the nearest
//! texel, keeps single noisy border texels from smearing into streaks.

use crate::cpu::{Coverage, Map};

/// Ring of texels never reached.
const UNREACHED: u16 = u16::MAX;

const NEIGHBORS: [(i64, i64); 8] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
];

/// The order in which padding fills texels, shared by all maps of a bake.
#[derive(Debug, Clone)]
pub struct Padding {
    width: u32,
    height: u32,
    /// Ring per texel: 0 for baked texels, [`UNREACHED`] beyond the padding.
    rings: Vec<u16>,
    /// Padding texels, ring by ring.
    order: Vec<u32>,
}

impl Padding {
    /// Plans the padding for a bake's `coverage` (missed texels count as
    /// baked). `None` extends the islands over the whole map.
    pub fn new(width: u32, height: u32, coverage: &[Coverage], padding: Option<u32>) -> Self {
        let mut rings = vec![UNREACHED; coverage.len()];
        let mut frontier: Vec<u32> = Vec::new();
        for (i, c) in coverage.iter().enumerate() {
            if *c != Coverage::Empty {
                rings[i] = 0;
                frontier.push(i as u32);
            }
        }
        let limit = padding.unwrap_or(u32::MAX).min(UNREACHED as u32 - 1);
        let mut order = Vec::new();
        let mut next = Vec::new();
        let mut ring = 0;
        while !frontier.is_empty() && ring < limit {
            ring += 1;
            for &i in &frontier {
                for j in neighbors(width, height, i as usize) {
                    if rings[j] == UNREACHED {
                        rings[j] = ring as u16;
                        next.push(j as u32);
                    }
                }
            }
            order.extend_from_slice(&next);
            std::mem::swap(&mut frontier, &mut next);
            next.clear();
        }
        Self {
            width,
            height,
            rings,
            order,
        }
    }

    /// Fills `map`'s padding texels.
    pub fn apply(&self, map: &mut Map) {
        assert_eq!((map.width, map.height), (self.width, self.height));
        let c = map.channels;
        let mut sum = vec![0.0f32; c];
        for &i in &self.order {
            let i = i as usize;
            let ring = self.rings[i];
            sum.fill(0.0);
            let mut count = 0;
            for j in neighbors(self.width, self.height, i) {
                if self.rings[j] < ring {
                    for (s, v) in sum.iter_mut().zip(&map.data[j * c..j * c + c]) {
                        *s += v;
                    }
                    count += 1;
                }
            }
            // Every padding texel was reached from an earlier ring.
            for (k, s) in sum.iter().enumerate() {
                map.data[i * c + k] = s / count as f32;
            }
        }
    }
}

/// Indices of the texels around texel `i`.
fn neighbors(width: u32, height: u32, i: usize) -> impl Iterator<Item = usize> {
    let (w, h) = (width as i64, height as i64);
    let (x, y) = (i as i64 % w, i as i64 / w);
    NEIGHBORS.iter().filter_map(move |&(dx, dy)| {
        let (nx, ny) = (x + dx, y + dy);
        (nx >= 0 && ny >= 0 && nx < w && ny < h).then(|| (ny * w + nx) as usize)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_with(width: u32, height: u32, texels: &[(u32, u32, f32)]) -> (Map, Vec<Coverage>) {
        let mut map = Map {
            width,
            height,
            channels: 1,
            data: vec![-1.0; (width * height) as usize],
        };
        let mut coverage = vec![Coverage::Empty; (width * height) as usize];
        for &(x, y, v) in texels {
            let i = (y * width + x) as usize;
            map.data[i] = v;
            coverage[i] = Coverage::Hit;
        }
        (map, coverage)
    }

    #[test]
    fn padding_grows_a_square_around_each_texel() {
        let (mut map, coverage) = map_with(9, 9, &[(4, 4, 7.0)]);
        Padding::new(9, 9, &coverage, Some(2)).apply(&mut map);
        for y in 0..9 {
            for x in 0..9 {
                let inside = (2..=6).contains(&x) && (2..=6).contains(&y);
                assert_eq!(
                    map.data[y * 9 + x],
                    if inside { 7.0 } else { -1.0 },
                    "({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn islands_grow_towards_each_other_and_none_fills_everything() {
        let (mut map, coverage) = map_with(10, 1, &[(1, 0, 1.0), (8, 0, 2.0)]);
        Padding::new(10, 1, &coverage, None).apply(&mut map);
        assert_eq!(map.data, [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
        // A texel reached from both sides at once takes their average.
        let (mut map, coverage) = map_with(3, 1, &[(0, 0, 1.0), (2, 0, 3.0)]);
        Padding::new(3, 1, &coverage, None).apply(&mut map);
        assert_eq!(map.data, [1.0, 2.0, 3.0]);
        // No padding leaves the background alone.
        let (mut map, coverage) = map_with(4, 1, &[(1, 0, 1.0)]);
        Padding::new(4, 1, &coverage, Some(0)).apply(&mut map);
        assert_eq!(map.data, [-1.0, 1.0, -1.0, -1.0]);
    }

    #[test]
    fn baked_texels_are_never_changed() {
        let (mut map, mut coverage) = map_with(3, 3, &[(0, 0, 1.0), (2, 2, 5.0)]);
        coverage[4] = Coverage::Missed;
        map.data[4] = 3.0;
        let before = map.data.clone();
        Padding::new(3, 3, &coverage, None).apply(&mut map);
        for i in [0, 4, 8] {
            assert_eq!(map.data[i], before[i]);
        }
        assert!(map.data.iter().all(|&v| v != -1.0));
    }
}
