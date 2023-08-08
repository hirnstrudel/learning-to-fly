use rand::Rng;

pub struct Network {
    layers: Vec<Layer>,
}

pub struct LayerTopology {
    pub neurons: usize,
}

#[derive(Clone)]
struct Layer {
    neurons: Vec<Neuron>,
}

#[derive(Clone)]
struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Network {
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(&inputs))
    }

    pub fn random(rng: &mut dyn rand::RngCore, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);

        let layers = layers
            .windows(2)
            .map(|layers| Layer::random(rng, layers[0].neurons, layers[1].neurons))
            .collect();

        Self { layers }
    }
}

impl Layer {
    pub fn propagate(&self, inputs: &[f32]) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }

    pub fn random(
        rng: &mut dyn rand::RngCore,
        input_neurons: usize,
        output_neurons: usize,
    ) -> Self {
        let neurons = (0..output_neurons)
            .map(|_| Neuron::random(rng, input_neurons))
            .collect();

        Self { neurons }
    }
}

impl Neuron {
    pub fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        (self.bias + output).max(0.0)
    }

    pub fn random(rng: &mut dyn rand::RngCore, output_size: usize) -> Self {
        let bias = rng.gen_range(-1.0..=1.0);

        let weights = (0..output_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        Self { bias, weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    mod neuron {
        use super::*;

        mod random {
            use super::*;
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;

            #[test]
            fn test() {
                let mut rng = ChaCha8Rng::from_seed(Default::default());
                let neuron = Neuron::random(&mut rng, 4);

                assert_relative_eq!(neuron.bias, -0.6255188);

                assert_relative_eq!(
                    neuron.weights.as_slice(),
                    [0.67383957, 0.8181262, 0.26284897, 0.5238807,].as_ref()
                );
            }
        }

        mod propagate {
            use super::*;

            #[test]
            fn test() {
                let neuron = Neuron {
                    bias: 0.5,
                    weights: vec![-0.3, 0.8],
                };

                assert_relative_eq!(neuron.propagate(&[-10.0, -10.0]), 0.0,);

                approx::assert_relative_eq!(
                    neuron.propagate(&[0.5, 1.0]),
                    (-0.3 * 0.5) + (0.8 * 1.0) + 0.5,
                );
            }
        }
    }

    mod layer {
        use super::*;

        mod random {
            use super::*;
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;

            #[test]
            fn test() {
                let mut rng = ChaCha8Rng::from_seed(Default::default());
                let layer = Layer::random(&mut rng, 3, 2);

                let actual: Vec<_> = layer
                    .neurons
                    .iter()
                    .map(|neuron| neuron.weights.as_slice())
                    .collect();

                let expected: Vec<&[f32]> = vec![
                    &[0.67383957, 0.8181262, 0.26284897],
                    &[-0.53516835, 0.069369674, -0.7648182],
                ];

                approx::assert_relative_eq!(actual.as_slice(), expected.as_slice());
            }
        }

        mod propagate {
            use super::*;

            #[test]
            fn test() {
                let neurons = vec![
                    Neuron {
                        bias: 0.1,
                        weights: vec![0.2, 0.3, 0.4],
                    },
                    Neuron {
                        bias: 0.5,
                        weights: vec![0.6, 0.7, 0.8],
                    },
                ];
                let input = vec![-0.5, 0.0, 0.5];

                let layer = Layer {
                    neurons: neurons.clone(),
                };

                let actual = layer.propagate(&input);
                let expected = vec![neurons[0].propagate(&input), neurons[1].propagate(&input)];

                assert_relative_eq!(actual.as_slice(), expected.as_slice());
            }
        }
    }

    mod network {
        use super::*;

        mod random {
            use super::*;
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;

            #[test]
            fn test() {
                let mut rng = ChaCha8Rng::from_seed(Default::default());

                let network = Network::random(
                    &mut rng,
                    &[
                        LayerTopology { neurons: 3 },
                        LayerTopology { neurons: 2 },
                        LayerTopology { neurons: 1 },
                    ],
                );

                assert_eq!(network.layers.len(), 2);
                assert_eq!(network.layers[0].neurons.len(), 2);

                approx::assert_relative_eq!(
                    network.layers[0].neurons[0].weights.as_slice(),
                    &[0.67383957, 0.8181262, 0.26284897].as_slice()
                );

                approx::assert_relative_eq!(
                    network.layers[0].neurons[1].weights.as_slice(),
                    &[-0.53516835, 0.069369674, -0.7648182].as_slice()
                );

                assert_eq!(network.layers[1].neurons.len(), 1);

                approx::assert_relative_eq!(
                    network.layers[1].neurons[0].weights.as_slice(),
                    &[-0.48879617, -0.19277132].as_slice()
                );
            }
        }

        mod propagate {
            use super::*;

            #[test]
            fn test() {
                let layers = vec![
                    Layer {
                        neurons: vec![
                            Neuron {
                                bias: 0.1,
                                weights: vec![0.2, 0.3, 0.4],
                            },
                            Neuron {
                                bias: 0.5,
                                weights: vec![0.6, 0.7, 0.8],
                            },
                        ],
                    },
                    Layer {
                        neurons: vec![Neuron {
                            bias: 0.2,
                            weights: vec![-0.5, 0.5],
                        }],
                    },
                ];

                let network = Network {
                    layers: layers.clone(),
                };

                let actual = network.propagate(vec![0.5, 0.6, 0.7]);
                let expected = layers[1].propagate(&layers[0].propagate(&[0.5, 0.6, 0.7]));

                assert_relative_eq!(actual.as_slice(), expected.as_slice());
            }
        }
    }
}
