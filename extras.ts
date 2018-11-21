import * as bm from "jsboxmuller";
import * as nj from "numjs";

export function zip<T1, T2>(x: T1[], y: T2[]): Array<[T1, T2]> {
    let result = [] as Array<[T1, T2]>;
    for (let i = 0; i < x.length; i++) {
        result.push([x[i], y[i]]);
    }
    return result;
}

export function* range(start: number, stop?: number, step: number = 1): Iterable<number> {
    if (stop) {
        for (let i = start; i < stop; i += step) {
            yield i;
        }
    } else {
        let i = start;
        while (true) {
            yield i;
            i += step;
        }
    }
}

export function argmax(array: number[]): number {
  if (array.length === 0) {
      return -1;
  }
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

/**
 * Generates n random numbers drawn from a normal distribution N(0, 1)
 *
 * @param n number of random numbers to generate
 */
function* generateRandomNumbers(n: number): Iterable<number> {
    let count = 0;
    while (n > count++) {
      yield bm() as number;
    }
}

/**
 * Return an x-by-y dimensional NDArray filled with random numbers
 * drawn from a normal distribution with mean 0 and unit variance
 */
export function randn(x, y): nj.NdArray<number> {
    let a = nj.array([...generateRandomNumbers(x * y)], "float32");
    return (a as any).reshape(x, y);
}
