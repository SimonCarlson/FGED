use crate::Vector3D;

use std::ops::Index;

pub struct Matrix3D {
    n: [[f64; 3]; 3],
}

impl Matrix3D {
    pub fn new(n00: f64, n01: f64, n02: f64,
        n10: f64, n11: f64, n12: f64,
        n20: f64, n21: f64, n22: f64) -> Self {
            let n = [[n00, n10, n20], [n01, n11, n21], [n02, n12, n22]];
            Self { n }
    }

    pub fn from_vector(a: Vector3D, b: Vector3D, c: Vector3D) -> Self {
        let n = [[a.x, a.y, a.z], [b.x, b.y, b.z], [c.x, c.y, c.z]];
        Self { n }
    }

    pub fn vector(&self, j: usize) -> Vector3D {
        let [x, y, z] = self.n[j];
        Vector3D { x, y, z }
    }
}

impl Index<[usize; 2]> for Matrix3D {
    type Output = f64;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
       &self.n[index[1]][index[0]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor() {
        let matrix = Matrix3D::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
        assert_eq!(matrix.n[0][0], 0.1);
        assert_eq!(matrix.n[0][1], 0.4);
        assert_eq!(matrix.n[0][2], 0.7);
        assert_eq!(matrix.n[1][0], 0.2);
        assert_eq!(matrix.n[1][1], 0.5);
        assert_eq!(matrix.n[1][2], 0.8);
        assert_eq!(matrix.n[2][0], 0.3);
        assert_eq!(matrix.n[2][1], 0.6);
        assert_eq!(matrix.n[2][2], 0.9);
    }

    #[test]
    fn vector_constructor() {
        let vector1 = Vector3D::new(0.1, 0.2, 0.3);
        let vector2 = Vector3D::new(0.4, 0.5, 0.6);
        let vector3 = Vector3D::new(0.7, 0.8, 0.9);
        let matrix = Matrix3D::from_vector(vector1, vector2, vector3);
        assert_eq!(matrix.n[0][0], 0.1);
        assert_eq!(matrix.n[0][1], 0.2);
        assert_eq!(matrix.n[0][2], 0.3);
        assert_eq!(matrix.n[1][0], 0.4);
        assert_eq!(matrix.n[1][1], 0.5);
        assert_eq!(matrix.n[1][2], 0.6);
        assert_eq!(matrix.n[2][0], 0.7);
        assert_eq!(matrix.n[2][1], 0.8);
        assert_eq!(matrix.n[2][2], 0.9);
    }

    #[test]
    fn index() {
        let matrix = Matrix3D::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
        assert_eq!(matrix[[0,0]], 0.1);
        assert_eq!(matrix[[0,1]], 0.2);
        assert_eq!(matrix[[0,2]], 0.3);
        assert_eq!(matrix[[1,0]], 0.4);
        assert_eq!(matrix[[1,1]], 0.5);
        assert_eq!(matrix[[1,2]], 0.6);
        assert_eq!(matrix[[2,0]], 0.7);
        assert_eq!(matrix[[2,1]], 0.8);
        assert_eq!(matrix[[2,2]], 0.9);
    }

    #[test]
    fn vector_index() {
        let vector1 = Vector3D::new(0.1, 0.2, 0.3);
        let vector2 = Vector3D::new(0.4, 0.5, 0.6);
        let vector3 = Vector3D::new(0.7, 0.8, 0.9);
        let matrix = Matrix3D::from_vector(vector1, vector2, vector3);
        assert_eq!(matrix.vector(0)[0], 0.1);
        assert_eq!(matrix.vector(0)[1], 0.2);
        assert_eq!(matrix.vector(0)[2], 0.3);
        assert_eq!(matrix.vector(1)[0], 0.4);
        assert_eq!(matrix.vector(1)[1], 0.5);
        assert_eq!(matrix.vector(1)[2], 0.6);
        assert_eq!(matrix.vector(2)[0], 0.7);
        assert_eq!(matrix.vector(2)[1], 0.8);
        assert_eq!(matrix.vector(2)[2], 0.9);
    }

}