use crate::vector::Vector3D;

use std::ops::{Add, Sub};

#[derive(Debug, PartialEq)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub fn distance_point_line(q: Point3D, p: Point3D, v: Vector3D) -> f64 {
    let a = (q - p).cross(&v);
    (a.dot(&a) / v.dot(&v)).sqrt()
}

pub fn distance_line_line(p1: Point3D, v1: Vector3D, p2: Point3D, v2: Vector3D) -> f64 {
    let dp = p2 - p1;

    let v12 = v1.dot(&v1);
    let v22 = v2.dot(&v2);
    let v1v2 = v1.dot(&v2);

    let det = v1v2 * v1v2 - v12 * v22;
    if det.abs() > f64::EPSILON {
        let inv_det = 1.0 / det;
        let dpv1 = dp.dot(&v1);
        let dpv2 = dp.dot(&v2);
        let t1 = (v1v2 * dpv2 - v22 * dpv1) * inv_det;
        let t2 = (v12 * dpv2 - v1v2 * dpv1) * inv_det;

        (dp + v2 * t2 - v1 * t1).magnitude()
    } else {
        let a = dp.cross(&v1);
        (a.dot(&a) / v12).sqrt()
    }
}

impl Point3D {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

impl Add<Vector3D> for Point3D {
    type Output = Point3D;
    fn add(self, rhs: Vector3D) -> Self::Output {
        Point3D::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub<Self> for Point3D {
    type Output = Vector3D;
    fn sub(self, rhs: Self) -> Self::Output {
        Vector3D::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

#[cfg(test)]
mod point3d_tests {
    use super::*;

    #[test]
    fn vector_point_addition() {
        let p = Point3D::new(2.0, 3.0, 4.0);
        let v = Vector3D::new(5.0, 6.0, 7.0);
        let expected = Point3D::new(7.0, 9.0, 11.0);
        assert_eq!(expected, p + v);
    }

    #[test]
    fn subtraction() {
        let p = Point3D::new(5.0, 6.0, 7.0);
        let q = Point3D::new(2.0, 3.0, 4.0);
        let expected = Vector3D::new(3.0, 3.0, 3.0);
        assert_eq!(expected, p - q);
    }
}