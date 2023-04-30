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

````html
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
````

## YouTube course - Machine Learning for Web Developers (Web ML)

https://www.youtube.com/playlist?list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui

Example:

3.1 Pre-trained models - easy set up: https://youtu.be/iTlj3gMYzw8?t=84

3.2 Selecting an ML model to use https://www.youtube.com/watch?v=MxgtqbPRjag&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=10
- problem -> consider a variety of existing model types -> narrow down models by comparison and user needs -> inference speed, file size, runtime memory usage from documentation or from DevTools analysis; and also usage environment (internet speed, device, etc.)

3.5 note: don't forget to dispose of memory! `input.dispose()` and `result.dispose()` but also more surprisingly `model.dispose()` https://www.youtube.com/watch?v=5QAO0mKFAKE&list=PLOU2XLYxmsILr3HQpqjLAUkIPa5EaZiui&index=18

## Quickly use pre-trained TensorFlow.js models in the browser

For example, it's only a few lines of code to do Q&A of text using MobileBERT: https://github.com/tensorflow/tfjs-models/tree/master/qna

Live demo: https://storage.googleapis.com/tfjs-models/demos/mobilebert-qna/index.html
