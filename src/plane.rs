use crate::{Vector3D, Point3D, Transform4D};

pub struct Plane {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Plane {
    pub fn dot_point(&self, p: Point3D) -> f64 {
        self.x * p.x + self.y * p.y + self.z * p.z + self.w
    }

    pub fn dot_vector(&self, v: Vector3D) -> f64 {
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
}