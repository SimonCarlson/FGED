use crate::{Vector3D, Transform4D, Matrix3D};

#[derive(Debug)]
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

    pub fn transform(self, h: &Transform4D) -> Line {
        let h0 = Vector3D::from(h[0]);
        let h1 = Vector3D::from(h[1]);
        let h2 = Vector3D::from(h[2]);

        let adj = Matrix3D::from_vector(h1.cross(&h2), h2.cross(&h0), h0.cross(&h1));
        let t = h.get_translation();
        let t_vec = Vector3D::new(t.x, t.y, t.z);
        let v = *h * self.direction;
        let m = adj * self.moment + t_vec.cross(&v);
        Line { direction: v, moment: m }
    }
}

#[cfg(test)]
mod line_tests {
    use super::*;

    #[test]
    fn transform() {
        let l = Line::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let t = Transform4D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
        let l1 = l.transform(&t);
        println!("{:?}", l1);
    }
}