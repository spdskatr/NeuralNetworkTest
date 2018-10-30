using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkTest
{
    public struct NeuralNetwork
    {
        // Store layers and score
        public Layer[] layers;
        public double score;

        // Blah blah constructors have nothing to do with ml
        public NeuralNetwork(NeuralNetwork copy)
        {
            layers = new Layer[copy.layers.Length];
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(copy.layers[i]);
            }
            score = copy.score;
        }
        public NeuralNetwork(int[] blueprint) : this(blueprint, null)
        {
        }
        public NeuralNetwork(int[] blueprint, ActivationFunction func)
        {
            // Make everything
            score = 1;
            layers = new Layer[blueprint.Length - 1];
            for (int i = 0; i < blueprint.Length - 1; i++)
            {
                layers[i] = new Layer(blueprint[i], blueprint[i + 1], func);
            }
        }

        /// <summary>
        /// Trains and finds the deltas for each training example, then takes an average for all the deltas and descends gradient for a set number of iterations.
        /// </summary>
        /// <param name="features">The input values. Should be an array [num_training_examples, num_inputs]</param>
        /// <param name="labels">The output values. Should be an array [num_training_examples, num_outputs]</param>
        /// <param name="iterations">The number of tierations to train it for.</param>
        /// <param name="learningRate">The learning rate for the NN. Faster is not always better.</param>
        public void Fit(double[,] features, double[,] labels, int iterations, double learningRate)
        {
            // TODO: Error check
            // Train for set number of iterations
            for (int a = 0; a < iterations; a++)
            {
                // Declare final average scores
                double aggregateScore = 0;
                // Shape: [layer_count][(shape of weights in said layer)]
                double[][,] aggregateWeightsDelta = new double[layers.Length][,];
                // Shape: [layer_count][(number of outputs in said layer)]
                double[][] aggregateBiasesDelta = new double[layers.Length][];

                // Initialise arrays
                for (int i = 0; i < layers.Length; i++)
                {
                    aggregateWeightsDelta[i] = new double[layers[i].weights.GetLength(0), layers[i].weights.GetLength(1)];
                    aggregateBiasesDelta[i] = new double[layers[i].biases.Length];
                }
                // Find deltas and output averages to aggregate 
                for (int i = 0; i < features.GetLength(0); i++)
                {
                    // Inputs and outputs for 1 training example
                    double[] testingFeatures = new double[features.GetLength(1)];
                    double[] testingLabels = new double[labels.GetLength(1)];

                    // Transcribing inputs and outputs
                    for (int j = 0; j < features.GetLength(1); j++)
                    {
                        testingFeatures[j] = features[i, j];
                    }
                    for (int j = 0; j < labels.GetLength(1); j++)
                    {
                        testingLabels[j] = labels[i, j];
                    }
                    // Feed Forward, Backprop, find deltas
                    FeedForward(testingFeatures);
                    Backprop(testingLabels);
                    // Work out score
                    aggregateScore += score / features.GetLength(0);
                    // Add averages of deltas to aggregate
                    for (int j = 0; j < layers.Length; j++)
                    {
                        double[,] weightsDelta = layers[j].GetWeightsDelta();
                        double[] biasesDelta = layers[j].GetBiasesDelta();
                        for (int x = 0; x < aggregateWeightsDelta[j].GetLength(0); x++)
                        {
                            for (int y = 0; y < aggregateWeightsDelta[j].GetLength(1); y++)
                            {
                                aggregateWeightsDelta[j][x, y] += weightsDelta[x, y] / features.GetLength(0);
                            }
                            aggregateBiasesDelta[j][x] += biasesDelta[x] / features.GetLength(0);
                        }
                    }
                }
                // Do gradient descent on the average of the deltas
                DescendGradient(learningRate, aggregateWeightsDelta, aggregateBiasesDelta);
                // Report
                if (a % 10000 == 0)
                {
                    Console.WriteLine("Score after {0} iterations: {1}", a, aggregateScore);
                }
            }
        }

        public void FeedForward(double[] inputs)
        {
            // TODO: Error check
            // Copy inputs to inputs of first layer
            Array.Copy(inputs, layers[0].inputs, inputs.Length);
            // For each layer, feed forward to next layer
            layers[0].FeedForward();
            for (int i = 1; i < layers.Length; i++)
            {
                Array.Copy(layers[i - 1].activations, layers[i].inputs, layers[i].inputs.Length);
                layers[i].FeedForward();
            }
        }

        public void Backprop(double[] expected)
        {
            // TODO: Error check
            // Calculate the error (Mean Squared Error)
            score = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                score += Math.Pow(layers[layers.Length - 1].activations[i] - expected[i], 2) / expected.Length;
            }
            // For each layer, calculate deltas
            layers[layers.Length - 1].Backprop(expected);
            for (int i = layers.Length - 2; i >= 0; i--)
            {
                layers[i].Backprop(layers[i + 1].weights, layers[i + 1].delta);
            }
        }

        public void DescendGradient(double learningRate, double[][,] weightsDelta, double[][] biasesDelta)
        {
            // For each layer, adjust gradient
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].DescendGradient(learningRate, weightsDelta[i], biasesDelta[i]);
            }
        }

        // Layer struct, does all the NN calculations
        // j = number of outputs
        // k = number of inputs
        public struct Layer
        {
            public double[] inputs; // shape: k
            public double[,] weights; // shape: j * k
            public double[] biases; // shape: j
            public double[] activations; // shape: j
            public double[] delta; // shape: j
            public ActivationFunction func;

            // Constructor stuff
            public Layer(Layer copy)
            {
                inputs = new double[copy.inputs.Length];
                weights = new double[copy.weights.GetLength(0), copy.weights.GetLength(1)];
                biases = new double[copy.biases.Length];
                activations = new double[copy.biases.Length];
                delta = new double[copy.biases.Length];
                func = copy.func;
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    for (int j = 0; j < weights.GetLength(1); j++)
                    {
                        weights[i, j] = copy.weights[i, j];
                    }
                    biases[i] = copy.biases[i];
                }
            }

            public Layer(int inputNeurons, int outputNeurons, ActivationFunction function)
            {
                inputs = new double[inputNeurons];
                weights = new double[outputNeurons, inputNeurons];
                biases = new double[outputNeurons];
                activations = new double[outputNeurons];
                delta = new double[outputNeurons];
                func = function;
                for (int i = 0; i < outputNeurons; i++)
                {
                    for (int j = 0; j < inputNeurons; j++)
                    {
                        weights[i, j] = Rand.Instance.NextDouble();
                    }
                    biases[i] = Rand.Instance.NextDouble();
                }
            }

            // "Inputs times weights, add bias, activate!" - Someone
            public void FeedForward()
            {
                // For each output
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    double result = 0;
                    // Inputs times weights
                    for (int j = 0; j < weights.GetLength(1); j++)
                    {
                        result += inputs[j] * weights[i, j];
                    }
                    // Add a bias, activate
                    activations[i] = func.Activate(result + biases[i]);
                }
            }

            // Delta = (output - expected) * derivative of activation function for output
            public void Backprop(double[] expectedOutput)
            {
                // For each output
                for (int i = 0; i < delta.Length; i++)
                {
                    delta[i] = (activations[i] - expectedOutput[i]) * func.ActivateDeriv(activations[i]); 
                }
            }

            // Delta = (sum of all (weights in next layer * delta of output for weights in next layer)) * derivative of activation function for output
            public void Backprop(double[,] weightsForward, double[] deltaForward)
            {
                // For each output
                for (int i = 0; i < delta.Length; i++)
                {
                    // Sum of all (weights in next layer * delta of output for weights in next layer)
                    double result = 0;
                    for (int j = 0; j < deltaForward.Length; j++)
                    {
                        result += weightsForward[j, i] * deltaForward[j];
                    }
                    // Times derivative of activation function for output
                    delta[i] = result * func.ActivateDeriv(activations[i]);
                }
            }

            // Delta for weight = delta for output of weight * value of input for weight
            public double[,] GetWeightsDelta()
            {
                double[,] result = new double[weights.GetLength(0), weights.GetLength(1)];
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    for (int j = 0; j < weights.GetLength(1); j++)
                    {
                        result[i, j] = inputs[j] * delta[i];
                    }
                }
                return result;
            }

            // Delta for bias = delta for output of bias
            public double[] GetBiasesDelta()
            {
                double[] result = new double[biases.Length];
                for (int i = 0; i < biases.Length; i++)
                {
                    result[i] = delta[i];
                }
                return result;
            }

            // Subtract deltas from all the weights and biases, with the rate learningRate
            public void DescendGradient(double learningRate, double[,] weightsDelta, double[] biasesDelta)
            {
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    for (int j = 0; j < weights.GetLength(1); j++)
                    {
                        weights[i, j] -= weightsDelta[i, j] * learningRate;
                    }
                    biases[i] -= biasesDelta[i] * learningRate;
                }
            }
        }
    }
}
