﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkTest
{
    public static class Rand
    {
        static Random random;
        public static Random Instance => (random = (random ?? new Random()));
    }
}
