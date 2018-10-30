using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkTest
{
    public abstract class ActivationFunction
    {
        public abstract double Activate(double input);
        public abstract double ActivateDeriv(double output);
    }
    public class SigmoidActivationFunction : ActivationFunction
    {
        public override double Activate(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-input));
        }
        public override double ActivateDeriv(double output)
        {
            return output * (1 - output);
        }
    }
    public class TanhActivationFunction : ActivationFunction
    {
        public override double Activate(double input)
        {
            return Math.Tanh(input);
        }
        public override double ActivateDeriv(double output)
        {
            return 1 - output * output;
        }
    }
}
