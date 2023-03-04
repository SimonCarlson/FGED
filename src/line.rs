use crate::{Vector3D, Transform4D, Matrix3D};

pub struct Line {
    pub direction: Vector3D,
    pub moment: Vector3D,
}

impl Line {
    pub fn new(vx: f64, vy: f64, vz: f64, mx: f64, my: f64, mz: f64) -> Self {
        Self { direction: Vector3D::new(vx, vy, vz), 
            moment: Vector3D::new(mx, my, mz) }
    }

    pub fn from_vector(v: Vector3D, m: Vector3D) -> Self {
        Self { direction: v, moment: m }
    }

    // pub fn transform(self, h: &Transform4D) -> Line {
        // let adj = Matrix3D::from_vector(h[1].cross(h[2]), h[2].cross(h[0]), h[0].cross(h[1]));
        // let t = h.get_translation();
        // let v = h * self.direction;
        // let m = adj * self.moment + t.cross(&v);
        // Line { direction: v, moment: m }
    // }
}