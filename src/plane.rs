use crate::{Vector3D, Point3D};

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
}