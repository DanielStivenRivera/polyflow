//! Axis-aligned bounding boxes.

use glam::Vec3;

/// An axis-aligned bounding box. An empty box has `min > max`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// A box containing nothing; growing it by any point yields that point.
    pub const EMPTY: Self = Self {
        min: Vec3::INFINITY,
        max: Vec3::NEG_INFINITY,
    };

    /// Smallest box containing all `points`. Returns [`Aabb::EMPTY`] for no points.
    pub fn from_points(points: &[Vec3]) -> Self {
        points.iter().fold(Self::EMPTY, |b, &p| b.grow(p))
    }

    /// Returns the box extended to contain `p`.
    #[must_use]
    pub fn grow(self, p: Vec3) -> Self {
        Self {
            min: self.min.min(p),
            max: self.max.max(p),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.min.cmpgt(self.max).any()
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Radius of the sphere centered on [`Aabb::center`] that encloses the box.
    pub fn radius(&self) -> f32 {
        (self.max - self.min).length() * 0.5
    }

    pub fn corners(&self) -> [Vec3; 8] {
        let (a, b) = (self.min, self.max);
        [
            Vec3::new(a.x, a.y, a.z),
            Vec3::new(b.x, a.y, a.z),
            Vec3::new(a.x, b.y, a.z),
            Vec3::new(b.x, b.y, a.z),
            Vec3::new(a.x, a.y, b.z),
            Vec3::new(b.x, a.y, b.z),
            Vec3::new(a.x, b.y, b.z),
            Vec3::new(b.x, b.y, b.z),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_points() {
        assert!(Aabb::from_points(&[]).is_empty());
        let b = Aabb::from_points(&[Vec3::new(1.0, -2.0, 3.0), Vec3::new(-1.0, 4.0, 0.0)]);
        assert_eq!(b.min, Vec3::new(-1.0, -2.0, 0.0));
        assert_eq!(b.max, Vec3::new(1.0, 4.0, 3.0));
        assert!(!b.is_empty());
    }
}
