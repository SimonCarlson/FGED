use crate::{Vector3D, Point3D, Transform4D};

use std::ops::Mul;

pub struct Plane {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

pub fn intersect_line_plane(p: Point3D, v: Vector3D, f: Plane) -> Option<Point3D> {
    let fv = f.dot_vector(&v);
    if fv.abs() > f64::EPSILON {
        let q = p - v * (f.dot_point(&p) / fv);
        Some(q)
    } else {
        None
    }
}

pub fn intersect_three_planes(f1: Plane, f2: Plane, f3: Plane) -> Option<Point3D> {
    let n1 = f1.get_normal();
    let n2 = f2.get_normal();
    let n3 = f3.get_normal();

    let n1xn2 = n1.cross(&n2);
    let det = n1xn2.dot(&n3);
    if det.abs() > f64::EPSILON {
        let q = (n3.cross(&n2) * f1.w + n1.cross(&n3) * f2.w - n1xn2 * f3.w) / det;
        Some(Point3D::new(q.x, q.y, q.z))
    } else {
        None
    }
}

pub fn intersect_two_planes(f1: Plane, f2: Plane) -> Option<(Point3D, Vector3D)> {
    let n1 = f1.get_normal();
    let n2 = f2.get_normal();

    let v = n1.cross(&n2);
    let det = v.dot(&v);
    if det.abs() > f64::EPSILON {
        let p = (v.cross(&n2) * f1.w + n1.cross(&v) * f2.w) / det;
        Some((Point3D::new(p.x, p.y, p.z), v))
    } else {
        None
    }
}

impl Plane {
    pub fn dot_point(&self, p: &Point3D) -> f64 {
        self.x * p.x + self.y * p.y + self.z * p.z + self.w
    }

    pub fn dot_vector(&self, v: &Vector3D) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    pub fn from_vector(v: Vector3D, d: f64) -> Self {
        Self { x: v.x, y: v.y, z: v.z, w: d }
    }

    pub fn get_normal(&self) -> Vector3D {
        Vector3D::new(self.x, self.y, self.z)
    }

    pub fn new(x: f64, y: f64, z: f64, d: f64) -> Self {
        Self { x, y, z, w: d }
    }

    pub fn make_reflection(&self) -> Transform4D {
        let x = self.x * -2.0;
        let y = self.y * -2.0;
        let z = self.z * -2.0;
        let nxny = x * self.y;
        let nxnz = x * self.z;
        let nynz = y * self.z;

        Transform4D::new(x * self.x + 1.0, nxny, nxnz, x * self.w,
            nxny, y * self.y + 1.0, nynz, y * self.w,
            nxnz, nynz, z * self.z + 1.0, z * self.w)
    }
}

impl Mul<Transform4D> for Plane {
    type Output = Plane;
    fn mul(self, rhs: Transform4D) -> Self::Output {
        Plane::new(
            self.x * rhs[[0,0]] + self.y * rhs[[1,0]] + self.z * rhs[[2,0]],
            self.x * rhs[[0,1]] + self.y * rhs[[1,1]] + self.z * rhs[[2,1]],
            self.x * rhs[[0,2]] + self.y * rhs[[1,2]] + self.z * rhs[[2,2]],
            self.x * rhs[[0,3]] + self.y * rhs[[1,3]] + self.z * rhs[[2,3]] + self.w,
        )
    }
}

#[cfg(test)]
mod plane_tests {
    use super::*;

    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn reflection() {
        let f = Plane::new(0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0, 0.0);
        let p = Point3D::new(5.0, 4.0, 0.0);
        let m = f.make_reflection();
        let reflected_p = m * p;
        // @TODO: Implement iterator for Point3D
        assert_approx_eq!(reflected_p.x, -4.0);
        assert_approx_eq!(reflected_p.y, -5.0);
        assert_approx_eq!(reflected_p.z, 0.0);
    }

    #[test]
    fn line_plane_intersection() {
        let f = Plane::new(1.0, 1.0, 0.0, 0.0);
        let p = Point3D::new(3.0, 3.0, 0.0);
        let v = Vector3D::new(0.0, 0.0, 1.0);
        assert_eq!(None, intersect_line_plane(p, v, f));

        let f = Plane::new(1.0, 1.0, 0.0, 0.0);
        let p = Point3D::new(-3.0, -4.0, 0.0);
        let v = Vector3D::new(1.0, 1.0, 0.0);
        let expected = Point3D::new(0.5, -0.5, 0.0);
        assert_eq!(expected, intersect_line_plane(p, v, f).unwrap());
    }

    #[test]
    fn three_plane_intersection() {
        let f1 = Plane::new(1.0, 0.0, 0.0, 0.0);
        let f2 = Plane::new(0.0, 1.0, 0.0, 0.0);
        let f3 = Plane::new(0.0, 0.0, 1.0, 0.0);
        let o = Point3D::origin();
        assert_eq!(o, intersect_three_planes(f1, f2, f3).unwrap());

        let f1 = Plane::new(1.0, 0.0, 0.0, 5.0);
        let f2 = Plane::new(1.0, 0.0, 0.0, 0.0);
        let f3 = Plane::new(0.0, 0.0, 1.0, 0.0);
        assert_eq!(None, intersect_three_planes(f1, f2, f3));
    }

    #[test]
    fn two_plane_intersection() {
        let f1 = Plane::new(1.0, 0.0, 0.0, 0.0);
        let f2 = Plane::new(0.0, 1.0, 0.0, 0.0);
        let o = Point3D::origin();
        let v = Vector3D::new(0.0, 0.0, 1.0);
        assert_eq!((o, v), intersect_two_planes(f1, f2).unwrap());

        let f1 = Plane::new(1.0, 0.0, 0.0, 5.0);
        let f2 = Plane::new(1.0, 0.0, 0.0, 0.0);
        assert_eq!(None, intersect_two_planes(f1, f2));
    }
}