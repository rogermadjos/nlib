mod topology;
mod activation;

pub use topology::{Topology, Layer};
pub use activation::Activation;
use rand::prelude::*;
use rand_distr::StandardNormal;

pub struct NeuralNetwork {
    topology: Topology,
    pub weights: Vec<Vec<Vec<f32>>>,
    pub biases: Vec<Vec<f32>>
}

impl NeuralNetwork {
    pub fn new(topology: Topology) -> Self {
        let mut rng = thread_rng();

        NeuralNetwork {
            topology: topology.clone(),
            weights: topology
                .clone()
                .layers
                .iter()
                .map(|layer| {
                    let std_dev: f32 = (2.0 / (layer.inputs + layer.outputs + 1) as f32).sqrt();

                    (0..layer.outputs)
                        .map(|_|
                                (0..layer.inputs)
                                    .map(|_| rng.sample::<f32, _>(StandardNormal) * std_dev)
                                    .collect()
                            )
                        .collect()
                })
                .collect(),
            biases: topology
                .clone()
                .layers
                .iter()
                .map(|layer| {
                    let std_dev: f32 = (2.0 / (layer.inputs + layer.outputs + 1) as f32).sqrt();

                    (0..layer.outputs)
                        .map(|_| rng.sample::<f32, _>(StandardNormal) * std_dev)
                        .collect()
                })
                .collect()
        }
    }

    pub fn eval(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut outputs: Vec<f32> = input.clone();

        for (layer, (weights, biases)) in self.topology.layers.iter().zip(self.weights.iter().zip(self.biases.iter())) {
            outputs = weights
                .iter()
                .zip(biases.iter())
                .map(|(weights, bias)| calculate_output(&outputs, weights, bias, layer.activation))
                .collect();
        }

        outputs
    }

    pub fn train(&self, training_data: &Vec<(Vec<f32>, Vec<f32>)>) {
        let mut rng = thread_rng();
        let m = (training_data.len() as f32 * 0.5) as usize;

        for _ in 0..100 {
            let samples: Vec<&(Vec<f32>, Vec<f32>)> = training_data
                .choose_multiple(&mut rng, m)
                .collect();

            

            println!("{:#?}", samples);
        }

        println!("{:#?}", training_data);
    }
}

fn calculate_output(input: &Vec<f32>, weights: &Vec<f32>, bias: &f32, activation: Activation) -> f32 {
    let sum = input
        .iter()
        .zip(weights.iter())
        .fold(0.0, |acc, (i, w)| acc + (i * w)) + bias;

    activation.base(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation::LeakyReLU;

    #[test]
    fn neural_network() {
        let neuralnet = NeuralNetwork::new(Topology::input(2).layer(2, LeakyReLU).layer(1, LeakyReLU));

        assert_eq!(neuralnet.weights.len(), 2);
        let mut iter = neuralnet.weights.iter().zip(neuralnet.biases.iter());
        let (weights, biases) = iter.next().unwrap();
        assert_eq!(weights.len(), 2);
        assert_eq!(biases.len(), 2);
        for item in weights.iter() {
            assert_eq!(item.len(), 2);
        }
        let (weights, biases) = iter.next().unwrap();
        assert_eq!(weights.len(), 1);
        assert_eq!(biases.len(), 1);
        for item in weights.iter() {
            assert_eq!(item.len(), 2);
        }

        let output = neuralnet.eval(&vec![1.0, 1.0]);
        assert_eq!(output.len(), 1);

        let training_data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];

        neuralnet.train(&training_data);
    }

    #[test]
    fn calculate_output() {
        let input: Vec<f32> = vec![2.0, 5.0, -1.0];
        let weights: Vec<f32> = vec![1.0, 0.5, 4.0];
        let bias: f32 = 1.5;

        let result = super::calculate_output(&input, &weights, &bias, LeakyReLU);

        assert_eq!(result, 2.0);
    }
}
