import * as nj from "numjs";
import * as shuffle from "shuffle-array";
import {argmax, randn, zip} from "./extras";

function sigmoid(z: nj.NdArray<number>) {
    // return 1.0 / (1.0 + exp(-z))
    return nj.divide(nj.ones(z.shape, z.dtype),
                     ( nj.exp(nj.negative(z)).add(1.0) ));
}

/** the derivative of sigmoid */
function sigmoidPrime(z: nj.NdArray<number>) {
    return sigmoid(z).multiply(
                        nj.ones(z.shape)
                        .subtract(sigmoid(z)));
}

/**
 * Construct a feed-forward neural network.
 *
 * All neurons in each layer are fully interconnected to all neurons
 * in the next layer.
 *
 * @param sizes an array containing the number of neurons for each layer
 */
export class Network {

  public numLayers: number;
  public sizes: number[];
  public biases: nj.NdArray<number>[];
  public weights: nj.NdArray<number>[];

  constructor(sizes: number[]) {
      this.numLayers = sizes.length;
      this.sizes = sizes;
      this.biases = sizes.slice(1).map(y => randn(y, 1));
      this.weights = zip(sizes.slice(0, -1), sizes.slice(1))
                     .map(([x, y]) => randn(y, x));
  }

  /** return the output of the network if 'a' is input */
  public feedforward(a: nj.NdArray<number>) {
      return zip(this.biases, this.weights)
             .reduce((prevA, [b, w]) => {
                 return sigmoid(nj.dot(w, prevA).add(b));
             }, a);
  }

  /**
   * Train the neural network using mini-batch stochastic
   * gradient descent.  The "training_data" is a list of tuples
   * "[x, y]" representing the training inputs and the desired
   * outputs (as NdArrays). The other non-optional parameters are
   * self-explanatory.  If "test_data" is provided then the
   * network will be evaluated against the test data after each
   * epoch, and partial progress printed out.  This is useful for
   * tracking progress, but slows things down substantially.
   */
  public sgd(trainingData: [nj.NdArray<number>, nj.NdArray<number>][],
             epochs: number,
             miniBatchSize: number,
             eta: number,
             testData?: [nj.NdArray<number>, number][]) {

    let n = trainingData.length;
    for (let j = 0; j < epochs; j++) {
        shuffle(trainingData);
        let miniBatches: [nj.NdArray<number>, nj.NdArray<number>][][] = [];
        for (let k = 0; k < n; k += miniBatchSize) {
            miniBatches.push(trainingData.slice(k, k + miniBatchSize));
        }
        for (let miniBatch of miniBatches) {
            this.updateMiniBatch(miniBatch, eta);
        }
        if (testData) {
            console.log(`Epoch ${j}: ${this.evaluate(testData)} / ${testData.length}`);
        } else {
            console.log(`Epoch ${j} complete`);
        }
    }
  }

  /**
   * Update the network's weights and biases by applying gradient descent
   * using backpropagation to a single mini batch.
   *
   * @param miniBatch a list of tuples [x, y]
   * @param eta the learning rate
   */
  public updateMiniBatch(miniBatch: [nj.NdArray<number>, nj.NdArray<number>][], eta: number) {
      let nablaB = this.biases.map(b => nj.zeros(b.shape));
      let nablaW = this.weights.map(w => nj.zeros(w.shape));
      for (let [x, y] of miniBatch) {
          let [deltaNablaB, deltaNablaW] = this.backprop(x, y);
          nablaB = zip(nablaB, deltaNablaB).map(([nb, dnb]) => nb.add(dnb));
          nablaW = zip(nablaW, deltaNablaW).map(([nw, dnw]) => nw.add(dnw));
      }

      // let miniBatchLenArr = nj.zeros(miniBatch[0][0].shape).add(miniBatch.length);

      this.weights = zip(this.weights, nablaW).map(([w, nw]) => {
          // w' = w - (eta/miniBatch.length) * nw
          return w.subtract(nw.multiply(eta / miniBatch.length));
      });
      this.biases = zip(this.biases, nablaB).map(([b, nb]) => {
          // b' = b - (eta/miniBatch.length) * nb
          return b.subtract(nb.multiply(eta / miniBatch.length));
      });
  }

  public evaluate(testData: [nj.NdArray<number>, number][]) {
      let testResults = testData.map(([x, y]) => {
        return [argmax(this.feedforward(x).flatten<number>().tolist()), y];
      });
      return testResults.map(([x, y]) => +(x === y)).reduce((i, j) => i + j, 0);
  }

  /**
   * Return a tuple `[nabla_b, nabla_w]` representing the
   * gradient for the cost function C_x.  `nabla_b` and
   * `nabla_w` are layer-by-layer lists of NDArrays, similar
   * to `this.biases` and `this.weights`.
   * @param x input activations
   * @param y expected output activations
   */
  private backprop(x: nj.NdArray<number>, y: nj.NdArray<number>): [nj.NdArray<number>[], nj.NdArray<number>[]] {
      let nablaB = this.biases.map(b => nj.zeros(b.shape));
      let nablaW = this.weights.map(w => nj.zeros(w.shape));
      // feedforward
      let activation = x;
      let activations = [x]; // list to store all activations, layer-by-layer
      let zs: nj.NdArray<number>[] = []; // list to store all z vectors, layer-by-layer
      zip(this.biases, this.weights).forEach(([b, w]) => {
          let z = nj.dot(w, activation).add(b);
          zs.push(z);
          activation = sigmoid(z);
          activations.push(activation);
      });
      // backward pass
      let delta =
        this.costDerivative(activations[activations.length - 1], y)
        .multiply(sigmoidPrime(zs[zs.length - 1]));
      nablaB[nablaB.length - 1] = delta;
      nablaW[nablaW.length - 1] = nj.dot(delta, activations[activations.length - 2].transpose());
      // iterate over each 'inner' layer l from back to front
      for (let l = 2; l < this.numLayers; l++) {
          let z = zs[zs.length - l];
          let sp = sigmoidPrime(z);
          delta =
            nj.dot(this.weights[this.weights.length - l + 1].transpose(),
                   delta)
            .multiply(sp);
          nablaB[nablaB.length - l] = delta;
          nablaW[nablaW.length - l] =
            nj.dot(delta,
                   activations[activations.length - l - 1].transpose());
      }
      return [nablaB, nablaW];
  }

  /**
   * Return the vector of partial derivatives \partial C_x /
   * \partial a for the output activations.
   */
  private costDerivative(outputActivations: nj.NdArray<number>, y: nj.NdArray<number>) {
      return outputActivations.subtract(y);
  }

}
