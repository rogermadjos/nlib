use crate::activation::Activation;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Layer {
    pub inputs: usize,
    pub outputs: usize,
    pub activation: Activation
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topology {
    pub inputs: usize,
    last: usize,
    pub layers: Vec<Layer>
}

impl Topology {
    pub fn input(size: usize) -> Self {
	    Topology {
            inputs: size,
            last: size,
            layers: Vec::new()
        }
    }

    pub fn layer(mut self, size: usize, activation: Activation) -> Self {
        self.layers.push(Layer {
            inputs: self.last,
            outputs: size,
            activation
        });
        self.last = size;

        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology() {
        use self::Activation::{Tanh};

        let topology = Topology::input(2)
            .layer(2, Tanh)
            .layer(1, Tanh);

        let mut iter = topology.layers.iter().map(|&item| item);

        assert_eq!(iter.next(), Some(Layer {
            inputs: 2,
            outputs: 2,
            activation: Tanh
        }));
        assert_eq!(iter.next(), Some(Layer {
            inputs: 2,
            outputs: 1,
            activation: Tanh
        }));
	}
}
