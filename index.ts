import * as sm from "source-map-support";
import { loadDataStreaming } from "./mnist_loader";
import { Network } from "./network";

sm.install();

async function main() {
    let {training, validation, test} = await loadDataStreaming("full");

    console.log(`# training samples = ${training.length}`);
    console.log(`# validation samples = ${validation.length}`);
    console.log(`# test samples = ${test.length}`);

    let net = new Network([784, 30, 10]);

    console.log(`Init: ${net.evaluate(test)} / ${test.length}`);

    net.sgd(training, 30, 10, 3.0, test);
}

main().catch(e => console.error(e));
