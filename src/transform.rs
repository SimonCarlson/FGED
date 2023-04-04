use crate::{vector::Vector3D, Vector4D};
use crate::point::Point3D;

use std::ops::{Index, Mul};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transform4D {
    n: [Vector4D; 4],
}

impl Transform4D {
    pub fn new(n00: f64, n01: f64, n02: f64, n03: f64,
        n10: f64, n11: f64, n12: f64, n13: f64,
        n20: f64, n21: f64, n22: f64, n23: f64) -> Self {
            let n1 = Vector4D::new(n00, n01, n02, n03);
            let n2 = Vector4D::new(n10, n11, n12, n13);
            let n3 = Vector4D::new(n20, n21, n22, n23);
            let n4 = Vector4D::new(0.0, 0.0, 0.0, 1.0);
            Self { n: [n1, n2, n3, n4] }
    }

    pub fn from_vector(a: Vector3D, b: Vector3D, c: Vector3D, p: Point3D) -> Self {
            let n1 = Vector4D::new(a.x, a.y, a.z, 0.0);
            let n2 = Vector4D::new(b.x, b.y, b.z, 0.0);
            let n3 = Vector4D::new(c.x, c.y, c.z, 0.0);
            let n4 = Vector4D::new(p.x, p.y, p.z, 1.0);
            Self { n: [n1, n2, n3, n4] }
    }

    pub fn get_translation(&self) -> Point3D {
        Point3D::new(self[3][0], self[3][1], self[3][2])
    }

    pub fn inverse(&self) -> Option<Transform4D> {
        let a = Vector3D::new(self[0][0], self[1][0], self[2][0]);
        let b = Vector3D::new(self[0][1], self[1][1], self[2][1]);
        let c = Vector3D::new(self[0][2], self[1][2], self[2][2]);
        let d = Vector3D::new(self[0][3], self[1][3], self[2][3]);
        let mut s = a.cross(&b);
        let mut t = c.cross(&d);
        
        let product = s.dot(&c);
        if product == 0.0 {
            None
        } else {
            let inv_det = 1.0 / product;
            s = s * inv_det;
            t = t * inv_det;
            let v = c * inv_det;

            let r0 = b.cross(&v);
            let r1 = v.cross(&a);

            Some(Transform4D::new(
                r0.x, r0.y, r0.z, -b.dot(&t),
                r1.x, r1.y, r1.z, a.dot(&t),
                s.x, s.y, s.z, -d.dot(&s)))
        }
    }

    pub fn set_translation(&self, p: &Point3D) -> Transform4D{
        let n1 = self[0];
        let n2 = self[1];
        let n3 = self[2];
        let n4 = Vector4D::new(p.x, p.y, p.z, 1.0);
        Self { n: [n1, n2, n3, n4] }
    }
}

impl Index<usize> for Transform4D {
    type Output = Vector4D;
    fn index(&self, index: usize) -> &Self::Output {
        &self.n[index]
    }
}

impl Index<[usize; 2]> for Transform4D {
    type Output = f64;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let [i, j] = index;
        if i > 3 {
            panic!("Index {} out of range.", j);
        }
        if j > 3 {
            panic!("Index {} out of range.", i);
        }
        &self.n[j][i]
    }
}

impl Mul<Point3D> for Transform4D {
    type Output = Point3D;
    fn mul(self, rhs: Point3D) -> Self::Output {
        Point3D::new(
            self[0][0] * rhs.x + self[0][1] * rhs.y + self[0][2] * rhs.z + self[0][3],
            self[1][0] * rhs.x + self[1][1] * rhs.y + self[1][2] * rhs.z + self[1][3],
            self[2][0] * rhs.x + self[2][1] * rhs.y + self[2][2] * rhs.z + self[2][3],
        )
    }
}

impl Mul<Self> for Transform4D {
    type Output = Transform4D;
    fn mul(self, rhs: Self) -> Self::Output {
        Transform4D::new(
            self[0][0] * rhs[0][0] + self[0][1] * rhs[1][0] + self[0][2] * rhs[2][0],
            self[0][0] * rhs[0][1] + self[0][1] * rhs[1][1] + self[0][2] * rhs[2][1],
            self[0][0] * rhs[0][2] + self[0][1] * rhs[1][2] + self[0][2] * rhs[2][2],
            self[0][0] * rhs[0][3] + self[0][1] * rhs[1][3] + self[0][2] * rhs[2][3] + self[0][3],

            self[1][0] * rhs[0][0] + self[1][1] * rhs[1][0] + self[1][2] * rhs[2][0],
            self[1][0] * rhs[0][1] + self[1][1] * rhs[1][1] + self[1][2] * rhs[2][1],
            self[1][0] * rhs[0][2] + self[1][1] * rhs[1][2] + self[1][2] * rhs[2][2],
            self[1][0] * rhs[0][3] + self[1][1] * rhs[1][3] + self[1][2] * rhs[2][3] + self[1][3],

            self[2][0] * rhs[0][0] + self[2][1] * rhs[1][0] + self[2][2] * rhs[2][0],
            self[2][0] * rhs[0][1] + self[2][1] * rhs[1][1] + self[2][2] * rhs[2][1],
            self[2][0] * rhs[0][2] + self[2][1] * rhs[1][2] + self[2][2] * rhs[2][2],
            self[2][0] * rhs[0][3] + self[2][1] * rhs[1][3] + self[2][2] * rhs[2][3] + self[2][3],
        )
    }
}

impl Mul<Vector3D> for Transform4D {
    type Output = Vector3D;
    fn mul(self, rhs: Vector3D) -> Self::Output {
        Vector3D::new(
            self[0][0] * rhs.x + self[0][1] * rhs.y + self[0][2] * rhs.z,
            self[1][0] * rhs.x + self[1][1] * rhs.y + self[1][2] * rhs.z,
            self[2][0] * rhs.x + self[2][1] * rhs.y + self[2][2] * rhs.z,
        )
    }
}

#[cfg(test)]
mod transform4d_tests {
    use crate::Matrix4D;

    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn constructor() {
        let t = Transform4D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
        assert_eq!(t[0][0], 1.0);
        assert_eq!(t[0][1], 2.0);
        assert_eq!(t[0][2], 3.0);
        assert_eq!(t[0][3], 4.0);
        assert_eq!(t[1][0], 5.0);
        assert_eq!(t[1][1], 6.0);
        assert_eq!(t[1][2], 7.0);
        assert_eq!(t[1][3], 8.0);
        assert_eq!(t[2][0], 9.0);
        assert_eq!(t[2][1], 10.0);
        assert_eq!(t[2][2], 11.0);
        assert_eq!(t[2][3], 12.0);
        assert_eq!(t[3][0], 0.0);
        assert_eq!(t[3][1], 0.0);
        assert_eq!(t[3][2], 0.0);
        assert_eq!(t[3][3], 1.0);
    }

    #[test]
    fn index() {
        let t = Transform4D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
        assert_approx_eq!(t[0][2], 3.0);
        assert_approx_eq!(t[1][3], 8.0);
    }

    #[test]
    fn get_and_set_translation() {
        let t = Transform4D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
        let p = Point3D::new(0.0, 0.0, 0.0);
        assert_eq!(p, t.get_translation());
        let p1 = Point3D::new(2.0, 3.0, 4.0);
        let t1 = t.set_translation(&p1);
        assert_eq!(p1, t1.get_translation());
    }

    #[test]
    fn transform_inversion() {
        let t = Transform4D::new(1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
        let inverted_t = t.inverse().unwrap();
        let product = t * inverted_t;
        let identity_matrix = Matrix4D::identity();
        assert_eq!(product[0][0], identity_matrix[0][0]);
        assert_eq!(product[0][1], identity_matrix[0][1]);
        assert_eq!(product[0][2], identity_matrix[0][2]);
        assert_eq!(product[0][3], identity_matrix[0][3]);
        assert_eq!(product[1][0], identity_matrix[1][0]);
        assert_eq!(product[1][1], identity_matrix[1][1]);
        assert_eq!(product[1][2], identity_matrix[1][2]);
        assert_eq!(product[1][3], identity_matrix[1][3]);
        assert_eq!(product[2][0], identity_matrix[2][0]);
        assert_eq!(product[2][1], identity_matrix[2][1]);
        assert_eq!(product[2][2], identity_matrix[2][2]);
        assert_eq!(product[2][3], identity_matrix[2][3]);
        assert_eq!(product[3][0], identity_matrix[3][0]);
        assert_eq!(product[3][1], identity_matrix[3][1]);
        assert_eq!(product[3][2], identity_matrix[3][2]);
        assert_eq!(product[3][3], identity_matrix[3][3]);
    }
}