const assertPositiveInteger = (value, name) => {
  if (!Number.isInteger(value) || value < 1) {
    throw new TypeError(`${name} must be a positive integer`);
  }
};

const assertFiniteVector = (values, expectedLength, name) => {
  if (
    !Array.isArray(values) ||
    values.length !== expectedLength ||
    values.some((value) => !Number.isFinite(value))
  ) {
    throw new TypeError(
      `${name} must contain ${expectedLength} finite numbers`,
    );
  }
};

export const XOR_DATA = Object.freeze([
  Object.freeze({
    inputs: Object.freeze([0, 0]),
    expected: Object.freeze([0]),
  }),
  Object.freeze({
    inputs: Object.freeze([0, 1]),
    expected: Object.freeze([1]),
  }),
  Object.freeze({
    inputs: Object.freeze([1, 0]),
    expected: Object.freeze([1]),
  }),
  Object.freeze({
    inputs: Object.freeze([1, 1]),
    expected: Object.freeze([0]),
  }),
]);

export class NeuralNetwork {
  constructor({
    inputSize = 2,
    hiddenLayers = [2],
    outputSize = 1,
    learningRate = 0.8,
    random = Math.random,
  } = {}) {
    assertPositiveInteger(inputSize, "inputSize");
    assertPositiveInteger(outputSize, "outputSize");
    if (!Array.isArray(hiddenLayers) || hiddenLayers.length === 0) {
      throw new TypeError("hiddenLayers must contain at least one layer size");
    }
    hiddenLayers.forEach((size, index) =>
      assertPositiveInteger(size, `hiddenLayers[${index}]`),
    );
    if (!Number.isFinite(learningRate) || learningRate <= 0) {
      throw new TypeError("learningRate must be a positive finite number");
    }
    if (typeof random !== "function") {
      throw new TypeError("random must be a function");
    }

    this.layerSizes = [inputSize, ...hiddenLayers, outputSize];
    this.learningRate = learningRate;
    this.weights = [];
    this.biases = [];
    this.activations = this.layerSizes.map((size) => new Array(size).fill(0));

    for (let layer = 0; layer < this.layerSizes.length - 1; layer += 1) {
      const fromSize = this.layerSizes[layer];
      const toSize = this.layerSizes[layer + 1];
      const scale = Math.sqrt(2 / fromSize);
      this.weights.push(
        Array.from({ length: toSize }, () =>
          Array.from({ length: fromSize }, () => (random() * 2 - 1) * scale),
        ),
      );
      this.biases.push(new Array(toSize).fill(0));
    }
  }

  static sigmoid(value) {
    return 1 / (1 + Math.exp(-value));
  }

  static sigmoidDerivative(output) {
    return output * (1 - output);
  }

  forward(inputs) {
    assertFiniteVector(inputs, this.layerSizes[0], "inputs");
    this.activations[0] = [...inputs];

    for (let layer = 0; layer < this.weights.length; layer += 1) {
      const previous = this.activations[layer];
      this.activations[layer + 1] = this.weights[layer].map(
        (neuronWeights, neuron) => {
          const weightedSum = neuronWeights.reduce(
            (sum, weight, input) => sum + weight * previous[input],
            this.biases[layer][neuron],
          );
          return NeuralNetwork.sigmoid(weightedSum);
        },
      );
    }

    return [...this.activations.at(-1)];
  }

  predict(inputs) {
    return this.forward(inputs);
  }

  trainSample(inputs, expected) {
    assertFiniteVector(expected, this.layerSizes.at(-1), "expected");
    const output = this.forward(inputs);
    const deltas = new Array(this.weights.length);
    const outputLayer = this.weights.length - 1;

    deltas[outputLayer] = output.map(
      (activation, neuron) =>
        (expected[neuron] - activation) *
        NeuralNetwork.sigmoidDerivative(activation),
    );

    for (let layer = outputLayer - 1; layer >= 0; layer -= 1) {
      deltas[layer] = this.activations[layer + 1].map((activation, neuron) => {
        const downstream = this.weights[layer + 1].reduce(
          (sum, nextWeights, nextNeuron) =>
            sum + nextWeights[neuron] * deltas[layer + 1][nextNeuron],
          0,
        );
        return downstream * NeuralNetwork.sigmoidDerivative(activation);
      });
    }

    for (let layer = 0; layer < this.weights.length; layer += 1) {
      for (let neuron = 0; neuron < this.weights[layer].length; neuron += 1) {
        for (
          let input = 0;
          input < this.weights[layer][neuron].length;
          input += 1
        ) {
          this.weights[layer][neuron][input] +=
            this.learningRate *
            deltas[layer][neuron] *
            this.activations[layer][input];
        }
        this.biases[layer][neuron] += this.learningRate * deltas[layer][neuron];
      }
    }

    return (
      output.reduce(
        (sum, value, index) => sum + (expected[index] - value) ** 2,
        0,
      ) / output.length
    );
  }

  trainEpoch(data = XOR_DATA) {
    if (!Array.isArray(data) || data.length === 0) {
      throw new TypeError("data must contain at least one training example");
    }
    return (
      data.reduce(
        (sum, example) =>
          sum + this.trainSample(example.inputs, example.expected),
        0,
      ) / data.length
    );
  }

  loss(data = XOR_DATA) {
    if (!Array.isArray(data) || data.length === 0) {
      throw new TypeError("data must contain at least one training example");
    }
    return (
      data.reduce((sum, example) => {
        assertFiniteVector(
          example.expected,
          this.layerSizes.at(-1),
          "expected",
        );
        const output = this.predict(example.inputs);
        return (
          sum +
          output.reduce(
            (sampleLoss, value, index) =>
              sampleLoss + (example.expected[index] - value) ** 2,
            0,
          ) /
            output.length
        );
      }, 0) / data.length
    );
  }

  snapshot() {
    return {
      layerSizes: [...this.layerSizes],
      learningRate: this.learningRate,
      weights: this.weights.map((layer) => layer.map((neuron) => [...neuron])),
      biases: this.biases.map((layer) => [...layer]),
    };
  }
}

export const createSeededRandom = (seed = 1) => {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
};
