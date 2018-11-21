import * as fs from "fs";
import * as nj from "numjs";
import * as readline from "readline";
import {createGunzip} from "zlib";

type MNISTExample = [nj.NdArray<number>, nj.NdArray<number>];
type MNISTExampleInput = [nj.NdArray<number>, number];

interface MNIST {
  training: Array<MNISTExample>;
  validation: Array<MNISTExampleInput>;
  test: Array<MNISTExampleInput>;
}

/** read data in streaming fashion from a gzipped newline-delimited JSON file */
function loadNDJSON(filename: string): Promise<Array<MNISTExampleInput>> {
  return new Promise((resolve, reject) => {
    let lineReader = readline.createInterface({
      input: fs.createReadStream(filename).pipe(createGunzip()),
    });

    let entries = [];
    lineReader.on("line", (line) => {
      try {
        let [inputImg, result] = JSON.parse(line);
        entries.push([
          nj.array(inputImg, "float32"),
          result ]);
      } catch (e) {
        reject(e);
        lineReader.close();
      }
    });
    lineReader.on("close", () => {
      resolve(entries);
    });
  });
}

function vectorize(j: number): nj.NdArray<number> {
  let vec = nj.zeros([10, 1], "float32");
  vec.set(j, 0, 1.0);
  return vec;
}

export async function loadDataStreaming(suffix: "sample"|"full"): Promise<MNIST> {
  let [training, validation, test] = await Promise.all(
    [ "training",
      "validation",
      "test" ].map(s => loadNDJSON(`./data/mnist_${s}_${suffix}.ndjson.gz`)));

  return {
    test,
    training: training.map(([img, res]) => [img, vectorize(res)] as MNISTExample),
    validation,
  };
}
