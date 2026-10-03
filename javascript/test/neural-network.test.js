import assert from "node:assert/strict";
import test from "node:test";

import {
  NeuralNetwork,
  XOR_DATA,
  createSeededRandom,
} from "../src/neural-network.js";

test("validates topology, rate, and vectors", () => {
  assert.throws(() => new NeuralNetwork({ hiddenLayers: [] }), /at least one/);
  assert.throws(
    () => new NeuralNetwork({ learningRate: 0 }),
    /positive finite/,
  );

  const network = new NeuralNetwork({ random: createSeededRandom(1) });
  assert.throws(() => network.predict([1]), /2 finite numbers/);
  assert.throws(
    () => network.trainSample([0, 1], [Number.NaN]),
    /1 finite numbers/,
  );
});

test("computes a deterministic forward pass", () => {
  const network = new NeuralNetwork({ hiddenLayers: [2], random: () => 0.5 });
  network.weights = [
    [
      [0.5, -0.25],
      [0.3, 0.8],
    ],
    [[0.7, -1.1]],
  ];
  network.biases = [[0.1, -0.2], [0.05]];

  const hidden = [
    NeuralNetwork.sigmoid(0.1 + 0.5),
    NeuralNetwork.sigmoid(-0.2 + 0.3),
  ];
  const expected = NeuralNetwork.sigmoid(
    0.05 + 0.7 * hidden[0] - 1.1 * hidden[1],
  );

  assert.ok(Math.abs(network.predict([1, 0])[0] - expected) < 1e-12);
});

test("returns a detached snapshot with the expected shape", () => {
  const network = new NeuralNetwork({
    inputSize: 2,
    hiddenLayers: [3, 2],
    outputSize: 1,
    random: createSeededRandom(2),
  });
  const snapshot = network.snapshot();

  assert.deepEqual(snapshot.layerSizes, [2, 3, 2, 1]);
  assert.deepEqual(
    snapshot.weights.map((layer) => layer.map((neuron) => neuron.length)),
    [[2, 2, 2], [3, 3], [2]],
  );
  snapshot.weights[0][0][0] = 999;
  assert.notEqual(network.weights[0][0][0], 999);
});

test("learns XOR with a seeded 2-2-1 network", () => {
  const network = new NeuralNetwork({
    hiddenLayers: [2],
    learningRate: 1,
    random: createSeededRandom(42),
  });

  const initialLoss = network.loss(XOR_DATA);
  for (let epoch = 0; epoch < 20000; epoch += 1) {
    network.trainEpoch(XOR_DATA);
  }

  const predictions = XOR_DATA.map(({ inputs }) => network.predict(inputs)[0]);
  assert.ok(network.loss(XOR_DATA) < initialLoss * 0.05);
  assert.ok(predictions[0] < 0.15, `00 prediction was ${predictions[0]}`);
  assert.ok(predictions[1] > 0.85, `01 prediction was ${predictions[1]}`);
  assert.ok(predictions[2] > 0.85, `10 prediction was ${predictions[2]}`);
  assert.ok(predictions[3] < 0.15, `11 prediction was ${predictions[3]}`);
});
