//! Ray and point queries against single triangles.

use polyflow_core::glam::Vec3;

/// Ray–triangle intersection (Möller–Trumbore), two-sided. Returns
/// `(t, u, v)` where the hit point is `origin + t * dir`, equal to
/// `(1 - u - v) * a + u * b + v * c`.
#[inline]
pub fn intersect_ray(
    origin: Vec3,
    dir: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> Option<(f32, f32, f32)> {
    let e1 = b - a;
    let e2 = c - a;
    let p = dir.cross(e2);
    let det = e1.dot(p);
    // Relative threshold, so tiny (or huge) meshes behave the same.
    if det.abs() <= f32::EPSILON * e1.length_squared().max(e2.length_squared()) * dir.length() {
        return None;
    }
    let inv_det = 1.0 / det;
    let s = origin - a;
    let u = s.dot(p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = dir.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv_det;
    Some((t, u, v))
}

/// Closest point on triangle `abc` to `p` (Ericson, Real-Time Collision
/// Detection 5.1.5). Returns the point and its barycentric `(u, v)` for `b`
/// and `c`.
#[inline]
pub fn closest_point(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, f32, f32) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, 0.0, 0.0);
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, 1.0, 0.0);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, v, 0.0);
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, 0.0, 1.0);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, 0.0, w);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, 1.0 - w, w);
    }
    let denom = va + vb + vc;
    if denom == 0.0 || !denom.is_finite() {
        // Degenerate triangle: fall back to the closest of its vertices.
        let candidates = [(a, 0.0, 0.0), (b, 1.0, 0.0), (c, 0.0, 1.0)];
        return candidates
            .into_iter()
            .min_by(|x, y| x.0.distance_squared(p).total_cmp(&y.0.distance_squared(p)))
            .unwrap();
    }
    let v = vb / denom;
    let w = vc / denom;
    (a + ab * v + ac * w, v, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Vec3 = Vec3::ZERO;
    const B: Vec3 = Vec3::X;
    const C: Vec3 = Vec3::Y;

    #[test]
    fn ray_hits_from_both_sides() {
        let (t, u, v) = intersect_ray(Vec3::new(0.25, 0.25, 1.0), Vec3::NEG_Z, A, B, C).unwrap();
        assert!((t - 1.0).abs() < 1e-6 && (u - 0.25).abs() < 1e-6 && (v - 0.25).abs() < 1e-6);
        let (t, ..) = intersect_ray(Vec3::new(0.25, 0.25, -2.0), Vec3::Z, A, B, C).unwrap();
        assert!((t - 2.0).abs() < 1e-6);
    }

    #[test]
    fn ray_misses() {
        assert!(
            intersect_ray(Vec3::new(0.8, 0.8, 1.0), Vec3::NEG_Z, A, B, C).is_none(),
            "outside"
        );
        assert!(
            intersect_ray(Vec3::new(0.2, 0.2, 1.0), Vec3::X, A, B, C).is_none(),
            "parallel"
        );
        assert!(
            intersect_ray(Vec3::Z, Vec3::NEG_Z, A, A, C).is_none(),
            "degenerate"
        );
    }

    #[test]
    fn closest_point_regions() {
        let cases = [
            (Vec3::new(0.2, 0.2, 3.0), Vec3::new(0.2, 0.2, 0.0)), // face
            (Vec3::new(-1.0, -1.0, 0.0), A),                      // vertex a
            (Vec3::new(2.0, -0.5, 1.0), B),                       // vertex b
            (Vec3::new(-0.2, 3.0, 0.0), C),                       // vertex c
            (Vec3::new(0.5, -1.0, 0.0), Vec3::new(0.5, 0.0, 0.0)), // edge ab
            (Vec3::new(-1.0, 0.5, 0.0), Vec3::new(0.0, 0.5, 0.0)), // edge ac
            (Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.5, 0.5, 0.0)), // edge bc
        ];
        for (p, expected) in cases {
            let (q, u, v) = closest_point(p, A, B, C);
            assert!(
                q.distance(expected) < 1e-6,
                "{p} -> {q}, expected {expected}"
            );
            let from_bary = A * (1.0 - u - v) + B * u + C * v;
            assert!(from_bary.distance(q) < 1e-6);
        }
    }

    #[test]
    fn closest_point_on_degenerate_triangle_is_finite() {
        let (q, ..) = closest_point(Vec3::splat(1.0), Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
        assert_eq!(q, Vec3::ZERO);
        let (q, ..) = closest_point(Vec3::new(0.5, 1.0, 0.0), Vec3::ZERO, Vec3::X, Vec3::X * 2.0);
        assert!(q.is_finite() && q.distance(Vec3::new(0.5, 0.0, 0.0)) < 1e-6);
    }
}
