use crate::Vector3D;

use std::ops::{Add, Index, Mul, Sub};

#[derive(Copy, Clone, Debug, PartialEq)]
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

impl Add<Self> for Matrix3D {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
       Matrix3D::new(self[[0,0]] + rhs[[0,0]], self[[0,1]] + rhs[[0,1]], self[[0,2]] + rhs[[0,2]],
        self[[1,0]] + rhs[[1,0]], self[[1,1]] + rhs[[1,1]], self[[1,2]] + rhs[[1,2]],
        self[[2,0]] + rhs[[2,0]], self[[2,1]] + rhs[[2,1]], self[[2,2]] + rhs[[2,2]])
    }
}

impl Index<[usize; 2]> for Matrix3D {
    type Output = f64;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        if index[0] > 2 {
            panic!("Index {} out of range", index[0]);
        }
        if index[1] > 2 {
            panic!("Index {} out of range", index[1]);
        }
        &self.n[index[1]][index[0]]
    }
}

impl Mul<f64> for Matrix3D {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Matrix3D::new(self[[0,0]] * rhs, self[[0,1]] * rhs, self[[0,2]] * rhs,
        self[[1,0]] * rhs, self[[1,1]] * rhs, self[[1,2]] * rhs,
        self[[2,0]] * rhs, self[[2,1]] * rhs, self[[2,2]] * rhs)
    }
}

impl Mul<Matrix3D> for Matrix3D {
    type Output = Self;
    fn mul(self, rhs: Matrix3D) -> Self::Output {
        Matrix3D::new(
            self[[0,0]] * rhs[[0,0]] + self[[0,1]] * rhs[[1,0]] + self[[0,2]] * rhs[[2,0]],
            self[[0,0]] * rhs[[0,1]] + self[[0,1]] * rhs[[1,1]] + self[[0,2]] * rhs[[2,1]],
            self[[0,0]] * rhs[[0,2]] + self[[0,1]] * rhs[[1,2]] + self[[0,2]] * rhs[[2,2]],
            self[[1,0]] * rhs[[0,0]] + self[[1,1]] * rhs[[1,0]] + self[[1,2]] * rhs[[2,0]],
            self[[1,0]] * rhs[[0,1]] + self[[1,1]] * rhs[[1,1]] + self[[1,2]] * rhs[[2,1]],
            self[[1,0]] * rhs[[0,2]] + self[[1,1]] * rhs[[1,2]] + self[[1,2]] * rhs[[2,2]],
            self[[2,0]] * rhs[[0,0]] + self[[2,1]] * rhs[[1,0]] + self[[2,2]] * rhs[[2,0]],
            self[[2,0]] * rhs[[0,1]] + self[[2,1]] * rhs[[1,1]] + self[[2,2]] * rhs[[2,1]],
            self[[2,0]] * rhs[[0,2]] + self[[2,1]] * rhs[[1,2]] + self[[2,2]] * rhs[[2,2]])
    }
}

impl Mul<Vector3D> for Matrix3D {
    type Output = Vector3D;
    fn mul(self, rhs: Vector3D) -> Self::Output {
        Vector3D::new(self[[0,0]] * rhs.x + self[[0,1]] * rhs.y + self[[0,2]] * rhs.z,
            self[[1,0]] * rhs.x + self[[1,1]] * rhs.y + self[[1,2]] * rhs.z,
            self[[2,0]] * rhs.x + self[[2,1]] * rhs.y + self[[2,2]] * rhs.z) 
    }
}

impl Mul<Matrix3D> for f64 {
    type Output = Matrix3D;
    fn mul(self, rhs: Matrix3D) -> Self::Output {
        rhs * self
    }
}

impl Sub<Self> for Matrix3D {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
       Matrix3D::new(self[[0,0]] - rhs[[0,0]], self[[0,1]] - rhs[[0,1]], self[[0,2]] - rhs[[0,2]],
        self[[1,0]] - rhs[[1,0]], self[[1,1]] - rhs[[1,1]], self[[1,2]] - rhs[[1,2]],
        self[[2,0]] - rhs[[2,0]], self[[2,1]] - rhs[[2,1]], self[[2,2]] - rhs[[2,2]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

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

    #[test]
    fn matrix_addition() {
        let matrix1 = Matrix3D::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
        let matrix2 = matrix1 + matrix1;
        assert_approx_eq!(matrix2[[0,0]], 0.2);
        assert_approx_eq!(matrix2[[0,1]], 0.4);
        assert_approx_eq!(matrix2[[0,2]], 0.6);
        assert_approx_eq!(matrix2[[1,0]], 0.8);
        assert_approx_eq!(matrix2[[1,1]], 1.0);
        assert_approx_eq!(matrix2[[1,2]], 1.2);
        assert_approx_eq!(matrix2[[2,0]], 1.4);
        assert_approx_eq!(matrix2[[2,1]], 1.6);
        assert_approx_eq!(matrix2[[2,2]], 1.8);
    }

    #[test]
    fn matrix_subtraction() {
        let matrix1 = Matrix3D::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
        let matrix2 = Matrix3D::new(0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8);
        let matrix3 = matrix2 - matrix1;
        assert_approx_eq!(matrix3[[0,0]], 0.1);
        assert_approx_eq!(matrix3[[0,1]], 0.2);
        assert_approx_eq!(matrix3[[0,2]], 0.3);
        assert_approx_eq!(matrix3[[1,0]], 0.4);
        assert_approx_eq!(matrix3[[1,1]], 0.5);
        assert_approx_eq!(matrix3[[1,2]], 0.6);
        assert_approx_eq!(matrix3[[2,0]], 0.7);
        assert_approx_eq!(matrix3[[2,1]], 0.8);
        assert_approx_eq!(matrix3[[2,2]], 0.9);
    }

    #[test]
    fn scalar_multiplication() {
        let matrix1 = Matrix3D::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
        let matrix2 = matrix1 * 5.0;
        assert_approx_eq!(matrix2[[0,0]], 0.5);
        assert_approx_eq!(matrix2[[0,1]], 1.0);
        assert_approx_eq!(matrix2[[0,2]], 1.5);
        assert_approx_eq!(matrix2[[1,0]], 2.0);
        assert_approx_eq!(matrix2[[1,1]], 2.5);
        assert_approx_eq!(matrix2[[1,2]], 3.0);
        assert_approx_eq!(matrix2[[2,0]], 3.5);
        assert_approx_eq!(matrix2[[2,1]], 4.0);
        assert_approx_eq!(matrix2[[2,2]], 4.5);
        assert_eq!(matrix2, 5.0 * matrix1);
    }

    #[test]
    fn matrix_multiplication() {
        let matrix1 = Matrix3D::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
        let matrix2 = Matrix3D::new(0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8);
        let matrix3 = matrix1 * matrix2;
        let expected = Matrix3D::new(0.6, 0.72, 0.84, 1.32, 1.62, 1.92, 2.04, 2.52, 3.0);
        assert_approx_eq!(matrix3[[0,0]], expected[[0,0]]);
        assert_approx_eq!(matrix3[[0,1]], expected[[0,1]]);
        assert_approx_eq!(matrix3[[0,2]], expected[[0,2]]);
        assert_approx_eq!(matrix3[[1,0]], expected[[1,0]]);
        assert_approx_eq!(matrix3[[1,1]], expected[[1,1]]);
        assert_approx_eq!(matrix3[[1,2]], expected[[1,2]]);
        assert_approx_eq!(matrix3[[2,0]], expected[[2,0]]);
        assert_approx_eq!(matrix3[[2,1]], expected[[2,1]]);
        assert_approx_eq!(matrix3[[2,2]], expected[[2,2]]);
    }

    #[test]
    fn matrix_vector_multiplication() {
        let matrix = Matrix3D::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
        let vector1 = Vector3D::new(0.2, 0.4, 0.6);
        let vector2 = matrix * vector1;
        assert_approx_eq!(vector2[0], 0.1*0.2 + 0.2*0.4 + 0.3*0.6);
        assert_approx_eq!(vector2[1], 0.4*0.2 + 0.5*0.4 + 0.6*0.6);
        assert_approx_eq!(vector2[2], 0.7*0.2 + 0.8*0.4 + 0.9*0.6);
    }
}