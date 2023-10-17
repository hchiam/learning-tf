# TensorFlow.js

Run models in the browser.

<https://www.tensorflow.org/js>

## Text Similarity Test

[GitHub code with link to CodePen demo](https://github.com/hchiam/text-similarity-test)

## UMAP visualization powered by [`umap-js`](https://github.com/PAIR-code/umap-js#umap-js) and [`chart.js`](https://github.com/chartjs/Chart.js)

https://github.com/hchiam/learning-tfjs-umap

## t-SNE visualization powered by [`tfjs-tsne`](https://github.com/hchiam/tfjs-tsne) and [`tfjs-vis`](https://github.com/hchiam/tfjs-vis)

https://github.com/hchiam/learning-tfjs-tsne

## Basic **Voice/Sound** Detection

[My live demo](https://tfjs-glitch-starter-howard.glitch.me/) on Glitch.

[One-file code example](https://github.com/hchiam/learning-tf/blob/master/js/sound-control-example.html) on GitHub.

[Google Codelab walk-through](https://codelabs.developers.google.com/codelabs/tensorflowjs-audio-codelab) + [Glitch.com starter code](https://glitch.com/~tfjs-glitch-starter):

## Basic **Image** Recognition

[Demo](https://codepen.io/hchiam/pen/LYYRLzz) based on a [smashingmagazine.com blogtutorial](https://www.smashingmagazine.com/2019/09/machine-learning-front-end-developers-tensorflowjs).

## Object Detection with a **Chrome Extension**

Example using [TensorFlow.js in a Chrome extension](https://github.com/tensorflow/tfjs-examples/tree/master/chrome-extension).

## **Offline** Predictions in a Web App

[Dev.to](https://dev.to/dar5hak/implementing-machine-learning-for-the-offline-web-with-tensorflowjs-46i) blog tutorial.

## Further learning

https://www.tensorflow.org/resources/learn-ml/basics-of-tensorflow-for-js-development

https://www.coursera.org/professional-certificates/tensorflow-in-practice

## Coursera

Course: https://www.coursera.org/learn/browser-based-models-tensorflow?specialization=tensorflow-data-and-deployment

Course repo: https://github.com/https-deeplearning-ai/tensorflow-2-public

Course forum: https://community.deeplearning.ai/c/tf2/tf2-course-1/141

Course lecture notes: https://community.deeplearning.ai/t/tf-data-and-deployment-course-1-lecture-notes/61289/3

`tfjs` should use mostly familiar APIs similar to what Python uses in `tensorflow.keras`.

```html
<head>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
</head>
<body>
  <script>
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
    model.summary(); // 2 params b/c y=mx+b has both m and b = weight and bias

    // we don't have numpy in js, so instead of np.array we have to use tf.tensor2d:
    const xs = tf.tensor2d([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], [6, 1]); // size: 6 x 1
    const ys = tf.tensor2d([-3.0, -1.0, 2.0, 3.0, 5.0, 7.0], [6, 1]);

    doTraining(model).then(() => {
      alert(model.predict(tf.tensor2d([10], [1, 1])));
    });

    async function doTraining(model) {
      const history = await model.fit(xs, ys, {
        epochs: 500,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            console.log(`Epoch: ${epoch} Loss: ${logs.loss}`);
          },
        },
      });
    }
  </script>
</body>
```

## Quickly use pre-trained TensorFlow.js models in the browser

For example, it's only a few lines of code to do Q&A of text using MobileBERT: https://github.com/tensorflow/tfjs-models/tree/master/qna

Live demo: https://storage.googleapis.com/tfjs-models/demos/mobilebert-qna/index.html

Even more templates and live demos on Glitch: https://glitch.com/@TensorFlowJS/official-tutorials-templates

## Running TensorFlow Decision Forests models with TensorFlow.js

https://www.tensorflow.org/decision_forests/tf_df_in_tf_js
1) **train** and **export** model file with **Python** in **Google Colab**
2) **convert** model file to TensorFlow.js with **Python** in **Google Colab**
3) **use** the TensorFlow.js model in **JavaScript** on the **web**

https://www.youtube.com/watch?v=5qgk9QJ4rdQ
- to help interpret:
  - `tfdf.model_plotter.plot_model_in_colab(model)` to see the structure of one of the trees
  - `model.make_inspector().variable_importances()['MEAN_DECREASE_IN_ACCURACY']`

https://www.youtube.com/watch?v=6DlWndLbk90 and https://ydf.readthedocs.io/en/latest/intro_df.html
- decision forest avoids overfitting problem of one big decision tree
- decision forests are great for tabular data, with highly interpretable results compared to neural networks
- decision forests have less hyperparams = easier to train
- decision forests are great as a first technique to try
- decision forests are NOT great for images, texts, time series data, graph data (use a neural network instead)

## YouTube course - Machine Learning for Web Developers (Web ML)

https://www.youtube.com/playlist?list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui

### Examples:

(The following are highlights of notes for myself, given that I've already taken related Coursera courses https://github.com/hchiam/learning-tf/tree/main/my_coursera_notes)

- tip: search the tf js API before trying to do something manually, like one-hot encoding `tf.oneHot`: https://js.tensorflow.org/api/latest/

3.1 Pre-trained models - easy set up: https://youtu.be/iTlj3gMYzw8?t=84

3.2 Selecting an ML model to use https://www.youtube.com/watch?v=MxgtqbPRjag&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=10
- problem -> consider a variety of existing model types -> narrow down models by comparison and user needs -> inference speed, file size, runtime memory usage from documentation or from DevTools analysis; and also usage environment (internet speed, device, etc.)

3.5 note: to avoid memory leaks, don't forget to dispose of memory when you're done training or want to retrain! `input.dispose()` and `result.dispose()` but also more surprisingly `model.dispose()` https://www.youtube.com/watch?v=5QAO0mKFAKE&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=18
  - 4.4.2 note: `returnValue = tf.tidy(function() {})` can automatically dispose non-returned tensors for you, but the `function` you pass to `tf.tidy` cannot be `async` https://www.youtube.com/watch?v=_m_ih8lXLvU&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=26
  - 4.4.3 note: `console.log(tf.memory().numTensors)` to check if you `.dispose()`-ed everything https://www.youtube.com/watch?v=aJ2IM6iy8y0&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=27 also btw, here's a demo of how to create a model from scratch if you need something custom: https://glitch.com/edit/#!/single-neuron-multi-input-linear-regression

4.2 note: training data set (to train models) -> validation data set (to choose which model to use) -> testing data set (as a final step, not "seen" before, to avoid models _indirectly_ being "trained" on the validation data set) https://www.youtube.com/watch?v=e5jNQ5TeK6E&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=22

4.6.2 note: reminders https://www.youtube.com/watch?v=48GnPgVGUKs&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=31
  - **softmax** makes sure the layer's neurons' outputs add up to 1 (100%), which is great for classification (one-hot)
  - **adam** optimizer automatically changes the learning rate over time
  - **trial and error and experience** will guide you on the number of layers and neurons (and hence connections) to use
    - for a given problem and size
    - for your given problem
    - for the best weighing of factors to consider like model memory size, prediction accuracy, prediction speed, etc.
  - `model.predict()` expects a tensor with a batch dimension specified (batch size can be of 1), so use `.expandDims()`:
    ```js
    let output = model.predict(input.expandDims());
    ```
    - to then predict, "un-batch" by removing one dimension with `.squeeze()`, the opposite of `.expandDims()`

4.7.1/4.7.2 note: usually pair `.conv2d()` with `.maxPooling2d()` to extract patterns in a position-independent way, while keeping information, while improving performance, since convolutional layers tend to create a lot of connections https://www.youtube.com/watch?v=lfTHBA-qpXU&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=32 / https://www.youtube.com/watch?v=-GxpuDee-a0&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=33

4.7.2 note: dropout layers can help a neural network pick up on the patterns that always/commonly matter, instead of one-off patterns, and random drop-out is commonly set to 50% or 25% https://www.youtube.com/watch?v=-GxpuDee-a0&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=33

5.1 Transfer learning https://www.youtube.com/watch?v=x-YFBvSpqz4&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=34
  - search for models at https://tfhub.dev/s?deployment-format=tfjs&network-architecture=mobilenet-v3
    - "image feature vector" models are base models with their heads (final layers) already removed so you can add your own custom classification layers
      - then don't forget to `model.save`
  - `const prediction = model.predict(inputTensor.expandDims()).squeeze();`
    - (note: `inputTensor` could also be the output of a frozen base model you passed the actual input to for training the overall transfer model)
  - `const indexOfMostConfidentPrediction = prediction.argMax().arraySync();`
  - `const mostConfidentPrediction = prediction.arraySync()[indexOfMostConfidentPrediction];`
  - `window.requestAnimationFrame(predictionLoopCallback);` for better performance than a fixed `setTimeout` delay

5.1 general tips: https://www.youtube.com/watch?v=x-YFBvSpqz4&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=34
  - warm up large models for speed by passing `tf.zeros([1, ...])` through it once inside `tf.tidy`
  - `array.splice(0)` to empty the array

5.3 Using layers models for transfer learning https://www.youtube.com/watch?v=PN4asCDITNg&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=36
  - `const layer = model.getLayer('name_of_layer_from_model_summary');`
  - `layer.trainable = false; // layer.trainable = true;`
  - `const truncatedModel = tf.model({inputs: model.inputs, outputs: layer.output});`
  - `const combinedModel = tf.sequential(); combinedModel.add(baseModel); combinedModel.add(model); combinedModel.compile({optimizer:'adam',loss:'categoricalCrossEntropy'}); combinedModel.summary(); await combinedModel.save('downloads://my-model'); ...`
  - also a general note: `model.summary(null, null, customPrint);`

6.2 how to convert from an existing Python model to a JS model, and also tips https://www.youtube.com/watch?v=EODze80347w&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=38

6.3 note: for language model embeddings: `4thRoot(number of words) = number of dimensions to use` https://www.youtube.com/watch?v=KC3iHks7wFs&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=39

6.4.4 Using a natural language model: Comment spam detection - web sockets https://www.youtube.com/watch?v=bkcUhMn3rik&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=43

6.5 tip: use python model maker in colab to retrain an exiting model on csv data, e.g. to account for special edge case examples https://www.youtube.com/watch?v=qxe3pWqgOwQ&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=44

### Links to more to get started fast:

- get **boilerplates/templates** to **start faster**: https://codepen.io/topic/tensorflow or https://glitch.com/@TensorFlowJS
- get **examples**: https://github.com/tensorflow/tfjs-examples
- get help: https://discuss.tensorflow.org/tag/tfjs
- get models: https://tensorflow.org/js/models
- get blog posts and docs and more: https://tensorflow.org/js
- get the latest library updates: https://github.com/tensorflow/tfjs
- see examples of realistic applications: https://goo.gle/made-with-tfjs
