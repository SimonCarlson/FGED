use crate::vector::Vector3D;

use std::ops::{Add, Sub};

#[derive(Debug, PartialEq)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
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