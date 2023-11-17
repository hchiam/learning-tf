# My (WORKING) demo of running a TensorFlow decision forest model with TensorFlow.js

Live demo: https://decision-forest-demo.surge.sh/

Created by combining

https://github.com/hchiam/decision-forests#usage-example

and

https://www.tensorflow.org/decision_forests/tutorials/beginner_colab

https://colab.research.google.com/drive/1y5JQxgP2eQKPno5rhWEP-pEiGgy1NByO#scrollTo=BRKLWIWNuOZ1

to make

https://colab.research.google.com/drive/1Fuh2NiHkhRGkq2F9GQTH0tY8JvZishay

1. I ran this code to get `tfjs_model.zip`: https://colab.research.google.com/drive/1Fuh2NiHkhRGkq2F9GQTH0tY8JvZishay (or see the back-up [.ipynb](https://github.com/hchiam/learning-tf/blob/main/js/decision-forest-demo-WORKING/short_tfdf_demo.ipynb)) (unzip the `.zip` file and get the `model.json`)
2. Open this file in a browser to try it out:
   - `bun js/decision-forest-demo-WORKING/index.tsx` or `npm run start` (you need to install `bun` first)
   - http://localhost:3000/
   - (see the console log output)
