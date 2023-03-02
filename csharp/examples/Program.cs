using System;
using System.Linq;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace HelloWorld
{
    class NeuralNetXorKeras
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            
            tf.enable_eager_execution();

            // // np.array isn't working for me for some reason
            var x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            var y = np.array(new float[,] { { 0 }, { 1 }, { 1 }, { 0 } });

            var model = keras.Sequential();
            model.add(keras.Input(2));
            model.add(keras.layers.Dense(32, keras.activations.Relu));
            model.add(keras.layers.Dense(1, keras.activations.Sigmoid));
            model.compile(optimizer: keras.optimizers.Adam(),
                loss: keras.losses.MeanSquaredError(),
                new[] { "accuracy" });
            model.fit(x, y, 1, 100);
            model.evaluate(x, y);
            Tensor result = model.predict(x, 4);
            
            Console.WriteLine(result.ToArray<float>() is [< 0.5f, > 0.5f, > 0.5f, < 0.5f]);
        }
    }
}