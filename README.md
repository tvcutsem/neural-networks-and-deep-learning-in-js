# Neural networks and deep learning in JS

This repository is a JavaScript port of the Python code in Michael Nielsen's
[Neural networks and deep learning](http://neuralnetworksanddeeplearning.com) book.

## Why?

Nielsen explains neural nets and deep learning from first principles,
using [working Python code](https://github.com/mnielsen/neural-networks-and-deep-learning)
to explain how neural networks work. Like
Nielsen, I learn ideas best by reading and writing executable code.
To force myself to understand everything, I ported the code from Python to JavaScript (to TypeScript actually).

I'm sharing the code in the hope that others may find it useful.
This code is written for didactic purposes only. If you're looking
for a practical implementation of deep learning in JavaScript,
check out [Tensorflow.js](https://js.tensorflow.org/).

Nielsen uses vanilla Python with the NumPy library for multi-dimensional
arrays and vector arithmetic. I chose to run with [numjs](https://github.com/nicolaspanel/numjs), which describes
itself as a "NumPy for JS". While numjs supports most of the features
needed to implement the code, I had to add a few utility methods.

I tried to stay faithful to the coding style in Nielsen's [original code](https://github.com/mnielsen/neural-networks-and-deep-learning).
The original code makes extensive use of Python list comprehensions.
The JavaScript code uses array higher-order functions like `map` instead.

## Getting started

I tested everything on node v10.13.0.

Clone this repo and install dependencies:

```
npm install
```

Compile TypeScript to JavaScript:

```
npm run build
```

Run the code (this trains a neural net that recognizes handwritten digits):

```
npm start
```

Example output:

```
Running node v10.13.0 (npm v6.4.1)
# training samples = 49999
# validation samples = 9999
# test samples = 9999
Epoch 0: 8227 / 9999
Epoch 1: 8316 / 9999
Epoch 2: 9294 / 9999
Epoch 3: 9319 / 9999
Epoch 4: 9326 / 9999
Epoch 5: 9369 / 9999
Epoch 6: 9407 / 9999
Epoch 7: 9357 / 9999
Epoch 8: 9392 / 9999
Epoch 9: 9378 / 9999
Epoch 10: 9420 / 9999
Epoch 11: 9399 / 9999
Epoch 12: 9427 / 9999
Epoch 13: 9436 / 9999
Epoch 14: 9436 / 9999
Epoch 15: 9411 / 9999
Epoch 16: 9437 / 9999
Epoch 17: 9454 / 9999
Epoch 18: 9436 / 9999
Epoch 19: 9436 / 9999
Epoch 20: 9472 / 9999
Epoch 21: 9443 / 9999
Epoch 22: 9457 / 9999
Epoch 23: 9438 / 9999
Epoch 24: 9458 / 9999
Epoch 25: 9452 / 9999
Epoch 26: 9468 / 9999
Epoch 27: 9462 / 9999
Epoch 28: 9448 / 9999
Epoch 29: 9433 / 9999
```

The code prints out the accuracy of the neural net on a test-set of 10K examples. The above network achieves a final classification accuracy on the test set of 94.33% (9433 out of 9999 examples classified correctly).

For details, read [chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html) of Nielsen's book.

## Performance

As mentioned above, this code was written for didactic purposes and is too slow for any practical use.

I was surpised to find that the JavaScript code runs quite a bit slower than the corresponding Python code. On my 2015 MBP, for the above training run of 30 epochs using a 3-layer neural net (784 x 30 x 10) I measured:

  * Python 2.7.6: 4m22.923s
  * Node 10.13.0: 29m42.972s

That's a factor 6.8x slowdown compared to the Python code.

I conjecture that this is mostly due to NumPy N-d arrays being far more optimized than NumJS N-d arrays (even though NumJS also uses native bindings to perform vector arithmetic).

## License

MIT