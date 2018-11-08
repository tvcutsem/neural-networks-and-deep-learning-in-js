# MNIST Dataset

The training data in Michael Nielsen's Neural networks and deep learning project
is stored as Python-pickled binary data, which is not easy to read into a
JavaScript project.

I created a simple conversion script to convert the pickled data into
[newline-delimited JSON](http://ndjson.org/).

In the process, I also split all the data into three separate files
(training, test, validation).

Each file contains a single `[image, digit]` sample per line
(this allows the JSON to be parsed incrementally, line-by-line).

Images are represented as 784-dimensional (= 28x28) arrays of `[float]` values (representing pixel intensity between `0.0` and `1.0`). Digits are represented
as integers `0` through `9`.

In Nielsen's code, the output digits are represented as 10-dimensional vectors in the training set and as decimal digits in the test and validation sets.
For consistency's sake I chose to change the training set format to be the same as the test and validation sets. After reading and parsing the training set file, the JS code transforms the digits of the training set into a 10-dimensional vector.

## Recreating the JSON data

To convert the data, copy `mnist_export_to_json.py` into Nielsen's [original
project](https://github.com/mnielsen/neural-networks-and-deep-learning) under `/src` and run `python mnist_export_to_json.py`.

This should generate three gzipped `.ndjson.gz` files containing
training, test and validation sets as newline-delimited JSON.
