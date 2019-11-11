#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    LeakyReLU,
    Tanh,
}

impl Activation {
    pub fn base(self, x: f32) -> f32 {
        use self::Activation::*;
        match self {
            ReLU      => relu(x),
            LeakyReLU => leaky_relu(x),
            Tanh      => tanh(x),
        }
    }

    pub fn derived(self, x: f32) -> f32 {
        use self::Activation::*;
        match self {
            ReLU      => relu_dx(x),
            LeakyReLU => leaky_relu_dx(x),
            Tanh      => tanh_dx(x),
        }
    }
}

fn relu(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { x }
}

fn relu_dx(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { 1.0 }
}

fn leaky_relu(x: f32) -> f32 {
    if x < 0.0 { 0.01 * x } else { x }
}

fn leaky_relu_dx(x: f32) -> f32 {
    if x < 0.0 { 0.01 } else { 1.0 }
}

pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

fn tanh_dx(x: f32) -> f32 {
    let fx = tanh(x);
    1.0 - fx * fx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu() {
        let act = Activation::ReLU;
        assert_eq!(act.base(-1.0), 0.0);
        assert_eq!(act.base(-0.5), 0.0);
        assert_eq!(act.base(0.0), 0.0);
        assert_eq!(act.base(0.5), 0.5);
        assert_eq!(act.base(1.0), 1.0);
        assert_eq!(act.derived(-1.0), 0.0);
        assert_eq!(act.derived(-0.5), 0.0);
        assert_eq!(act.derived(0.0), 1.0);
        assert_eq!(act.derived(0.5), 1.0);
        assert_eq!(act.derived(1.0), 1.0);
    }

    #[test]
    fn leaky_relu() {
        let act = Activation::LeakyReLU;
        assert_eq!(act.base(-1.0), -0.01);
        assert_eq!(act.base(-0.5), -0.005);
        assert_eq!(act.base(0.0), 0.0);
        assert_eq!(act.base(0.5), 0.5);
        assert_eq!(act.base(1.0), 1.0);
        assert_eq!(act.derived(-1.0), 0.01);
        assert_eq!(act.derived(-0.5), 0.01);
        assert_eq!(act.derived(0.0), 1.0);
        assert_eq!(act.derived(0.5), 1.0);
        assert_eq!(act.derived(1.0), 1.0);
    }

    #[test]
    fn tanh() {
        let act = Activation::Tanh;
        assert_eq!(act.base(-1.0), -0.7615942);
        assert_eq!(act.base(-0.5), -0.46211717);
        assert_eq!(act.base(0.0), 0.0);
        assert_eq!(act.base(0.5), 0.46211717);
        assert_eq!(act.base(1.0), 0.7615942);
        assert_eq!(act.derived(-1.0), 0.41997433);
        assert_eq!(act.derived(-0.5), 0.7864477);
        assert_eq!(act.derived(0.0), 1.0);
        assert_eq!(act.derived(0.5), 0.7864477);
        assert_eq!(act.derived(1.0), 0.41997433);
    }
}
