using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetworkTest
{
    public class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork net2 = new NeuralNetwork(new int[] { 2, 3, 1 }, new TanhActivationFunction());
            net2.FeedForward(new double[] { 0, 0 });
            Console.WriteLine("Output for input 0, 0: " + net2.layers[net2.layers.Length - 1].activations[0]);
            net2.FeedForward(new double[] { 0, 1 });
            Console.WriteLine("Output for input 0, 1: " + net2.layers[net2.layers.Length - 1].activations[0]);
            net2.FeedForward(new double[] { 1, 0 });
            Console.WriteLine("Output for input 1, 0: " + net2.layers[net2.layers.Length - 1].activations[0]);
            net2.FeedForward(new double[] { 1, 1 });
            Console.WriteLine("Output for input 1, 1: " + net2.layers[net2.layers.Length - 1].activations[0]);
            net2.Fit(new double[,]
            {
                { 0, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 1 }
            }, new double[,]
            {
                { 0 },
                { 1 },
                { 1 },
                { 0 }
            }, 60001, 1);
            net2.FeedForward(new double[] { 0, 0 });
            Console.WriteLine("Output for input 0, 0: " + net2.layers[net2.layers.Length - 1].activations[0]);
            net2.FeedForward(new double[] { 0, 1 });
            Console.WriteLine("Output for input 0, 1: " + net2.layers[net2.layers.Length - 1].activations[0]);
            net2.FeedForward(new double[] { 1, 0 });
            Console.WriteLine("Output for input 1, 0: " + net2.layers[net2.layers.Length - 1].activations[0]);
            net2.FeedForward(new double[] { 1, 1 });
            Console.WriteLine("Output for input 1, 1: " + net2.layers[net2.layers.Length - 1].activations[0]);
            Console.ReadKey();
        }
    }
}
