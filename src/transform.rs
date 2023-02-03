use crate::vector::Vector3D;
use crate::point::Point3D;

use std::ops::Index;

#[derive(Debug, PartialEq)]
pub struct Transform4D {
    n: [[f64; 4]; 4],
}

impl Transform4D {
    pub fn new(n00: f64, n01: f64, n02: f64, n03: f64,
        n10: f64, n11: f64, n12: f64, n13: f64,
        n20: f64, n21: f64, n22: f64, n23: f64) -> Self {
            let n = [[n00, n10, n20, 0.0], [n01, n11, n21, 0.0], [n02, n12, n22, 0.0], [n03, n13, n23, 1.0]];
            Self { n }
    }

    pub fn from_vector(a: Vector3D, b: Vector3D, c: Vector3D, p: Point3D) -> Self {
            let n = [[a.x, a.y, a.z, 0.0], [b.x, b.y, b.z, 0.0], [c.x, c.y, c.z, 0.0], [p.x, p.y, p.z, 1.0]];
            Self { n }
    }

    pub fn get_translation(&self) -> Point3D {
        Point3D::new(self[[3,0]], self[[3,1]], self[[3,2]])
    }

    // pub fn inverse(&self) -> Option<Transform4D> {
    //     let a = self[0];
    //     let b = self[1];
    //     let c = self[2];
    //     let d = self[3];
    //     let mut s = a.cross(&b);
    //     // let mut t = c.cross(&d);
        
    //     // let product = s.dot(&c);
    //     if product == 0.0 {
    //         None
    //     } else {
    //         let inv_det = 1.0 / product;
    //         s = s * inv_det;
    //         t = t * inv_det;
    //         let v = c * inv_det;

    //         let r0 = b.cross(&v);
    //         let r1 = v.cross(&a);

    //         Some(Transform4D::new(r0.x, r0.y, r0.z, -b.dot(&t),
    //             r1.x, r1.y, r1.z, a.dot(&t),
    //             s.x, s.y, s.z, -d.dot(&s)))
    //     }
    // }

    pub fn set_translation(&self, p: &Point3D) -> Transform4D{
            let n = [[self[[0,0]], self[[1,0]], self[[2,0]], p.x],
                    [self[[0,1]], self[[1,1]], self[[2,1]], p.y],
                    [self[[0,2]], self[[1,2]], self[[2,2]], p.z],
                    [self[[0,3]], self[[1,3]], self[[2,3]], 1.0]];
            Self { n }
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

// impl Index<usize> for Transform4D {
    // type Output = Vector3D;
    // fn index(&self, index: usize) -> Self::Output {
        // match index {
            // 0 => Vector3D::new(self[[0,0]], self[[0,1]], self[[0,2]]),
            // 1 => Vector3D::new(self[[1,0]], self[[1,1]], self[[1,2]]),
            // 2 => Vector3D::new(self[[2,0]], self[[2,1]], self[[2,2]]),
            // _ => panic!("Index {} out of range.", index),
        // }
    // }
// }

#[cfg(test)]
mod transform4d_tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn index() {
        let t = Transform4D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
        assert_approx_eq!(3.0, t[[0,2]]);
        assert_approx_eq!(8.0, t[[1,3]]);
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
}