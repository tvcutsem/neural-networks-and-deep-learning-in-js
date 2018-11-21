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

Example output for training MNIST on 50K training samples using a simple 784x30x10 feed-forward network:

```
Running node v10.13.0 (npm v6.4.1)
# training samples = 50000
# validation samples = 10000
# test samples = 10000
Init: 1147 / 10000
Epoch 0: 8993 / 10000
Epoch 1: 9143 / 10000
Epoch 2: 9264 / 10000
Epoch 3: 9281 / 10000
Epoch 4: 9343 / 10000
Epoch 5: 9348 / 10000
Epoch 6: 9333 / 10000
Epoch 7: 9417 / 10000
Epoch 8: 9328 / 10000
Epoch 9: 9429 / 10000
Epoch 10: 9407 / 10000
Epoch 11: 9398 / 10000
Epoch 12: 9428 / 10000
Epoch 13: 9429 / 10000
Epoch 14: 9455 / 10000
Epoch 15: 9466 / 10000
Epoch 16: 9476 / 10000
Epoch 17: 9444 / 10000
Epoch 18: 9457 / 10000
Epoch 19: 9466 / 10000
Epoch 20: 9464 / 10000
Epoch 21: 9482 / 10000
Epoch 22: 9445 / 10000
Epoch 23: 9469 / 10000
Epoch 24: 9480 / 10000
Epoch 25: 9453 / 10000
Epoch 26: 9463 / 10000
Epoch 27: 9473 / 10000
Epoch 28: 9470 / 10000
Epoch 29: 9461 / 10000
```

The code prints out the accuracy of the neural net on a test-set of 10K examples. The above network achieves a final classification accuracy on the test set of 94.61% (9461 out of 10000 examples classified correctly).

For details, read [chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html) of Nielsen's book.

## Performance

As mentioned above, this code was written for didactic purposes and is too slow for any practical use.

I was surpised to find that the JavaScript code runs quite a bit slower than the corresponding Python code. On my 2015 MBP, for the above training run of 30 epochs using a 3-layer neural net (784 x 30 x 10) I measured:

  * Python 2.7.6: 4m22.923s
  * Node 10.13.0: 29m42.972s

That's a factor 6.8x slowdown compared to the Python code.

I conjecture that this is mostly due to NumPy N-d arrays being far more optimized than NumJS N-d arrays (even though NumJS ndarrays also use [clever tricks](http://mikolalysenko.github.io/ndarray-presentation/) to efficiently represent vectors in JavaScript).

A small microbenchmark seems to confirm this: running a tight loop of
multiplying a 784x30 by a 30x1 ndarray 10K times takes on the order of 700 milliseconds in JS/numjs and on the order of 65 milliseconds in Python/numpy, a 10x performance difference.

## License

MIT