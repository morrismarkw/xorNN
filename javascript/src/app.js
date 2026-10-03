import { NeuralNetwork, XOR_DATA } from "./neural-network.js";

const elements = {
  hiddenLayers: document.querySelector("#hidden-layers"),
  neurons: document.querySelector("#neurons-per-layer"),
  rate: document.querySelector("#learning-rate"),
  rateOutput: document.querySelector("#rate-output"),
  train: document.querySelector("#train"),
  stop: document.querySelector("#stop"),
  reset: document.querySelector("#reset"),
  export: document.querySelector("#export"),
  epoch: document.querySelector("#epoch"),
  loss: document.querySelector("#loss"),
  predictions: document.querySelector("#predictions"),
  network: document.querySelector("#network"),
  lossChart: document.querySelector("#loss-chart"),
  status: document.querySelector("#status"),
};

let network;
let epoch = 0;
let lossHistory = [];
let training = false;
let latestActivations = [];

const svgElement = (name, attributes = {}) => {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) =>
    element.setAttribute(key, value),
  );
  return element;
};

const buildNetwork = () => {
  const hiddenCount = Number(elements.hiddenLayers.value);
  const neurons = Number(elements.neurons.value);
  network = new NeuralNetwork({
    hiddenLayers: new Array(hiddenCount).fill(neurons),
    learningRate: Number(elements.rate.value),
  });
  epoch = 0;
  lossHistory = [];
  latestActivations = network.activations.map((layer) => [...layer]);
  render();
  elements.status.textContent = "Network reset with new random weights.";
};

const layerPositions = () =>
  network.layerSizes.map((size, layer) => {
    const x = 70 + (620 * layer) / (network.layerSizes.length - 1);
    return Array.from({ length: size }, (_, neuron) => ({
      x,
      y: 62 + (270 * (neuron + 1)) / (size + 1),
    }));
  });

const renderNetwork = () => {
  elements.network.replaceChildren();
  const positions = layerPositions();

  network.weights.forEach((layerWeights, layer) => {
    layerWeights.forEach((neuronWeights, toNeuron) => {
      neuronWeights.forEach((weight, fromNeuron) => {
        const from = positions[layer][fromNeuron];
        const to = positions[layer + 1][toNeuron];
        elements.network.append(
          svgElement("line", {
            class: "connection",
            x1: from.x,
            y1: from.y,
            x2: to.x,
            y2: to.y,
            stroke: weight >= 0 ? "#57d39b" : "#ff706b",
            "stroke-width": Math.min(8, 0.7 + Math.abs(weight) * 1.8),
          }),
        );
      });
    });
  });

  positions.forEach((layerPositionsList, layer) => {
    const label = svgElement("text", {
      class: "layer-label",
      x: layerPositionsList[0].x,
      y: 28,
    });
    label.textContent =
      layer === 0
        ? "INPUT"
        : layer === positions.length - 1
          ? "OUTPUT"
          : `HIDDEN ${layer}`;
    elements.network.append(label);

    layerPositionsList.forEach((position, neuron) => {
      elements.network.append(
        svgElement("circle", {
          class: layer === positions.length - 1 ? "node output" : "node",
          cx: position.x,
          cy: position.y,
          r: 26,
        }),
      );
      const value = latestActivations[layer]?.[neuron] ?? 0;
      const text = svgElement("text", {
        class: "node-label",
        x: position.x,
        y: position.y,
      });
      text.textContent = value.toFixed(2);
      elements.network.append(text);
    });
  });
};

const renderPredictions = () => {
  elements.predictions.replaceChildren();
  XOR_DATA.forEach(({ inputs, expected }) => {
    const prediction = network.predict(inputs)[0];
    latestActivations = network.activations.map((layer) => [...layer]);
    const row = document.createElement("tr");
    const correct = (prediction >= 0.5 ? 1 : 0) === expected[0];
    [inputs[0], inputs[1], expected[0]].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    const result = document.createElement("td");
    result.className = correct ? "correct" : "incorrect";
    result.textContent = prediction.toFixed(4);
    row.append(result);
    elements.predictions.append(row);
  });
};

const renderLossChart = () => {
  elements.lossChart.replaceChildren();
  elements.lossChart.append(
    svgElement("line", {
      class: "chart-axis",
      x1: 35,
      y1: 10,
      x2: 35,
      y2: 190,
    }),
    svgElement("line", {
      class: "chart-axis",
      x1: 35,
      y1: 190,
      x2: 510,
      y2: 190,
    }),
  );
  if (lossHistory.length < 2) return;

  const sampled =
    lossHistory.length > 500
      ? lossHistory.filter(
          (_, index) => index % Math.ceil(lossHistory.length / 500) === 0,
        )
      : lossHistory;
  const maximum = Math.max(...sampled, 0.001);
  const points = sampled.map((value, index) => {
    const x = 35 + (475 * index) / Math.max(1, sampled.length - 1);
    const y = 190 - (175 * value) / maximum;
    return `${x},${y}`;
  });
  elements.lossChart.append(
    svgElement("polyline", { class: "chart-line", points: points.join(" ") }),
  );
};

const render = () => {
  renderPredictions();
  renderNetwork();
  renderLossChart();
  const currentLoss = network.loss(XOR_DATA);
  elements.epoch.textContent = epoch.toLocaleString();
  elements.loss.textContent = currentLoss.toFixed(6);
  elements.rateOutput.textContent = Number(elements.rate.value).toFixed(2);
};

const finishTraining = (message) => {
  training = false;
  elements.train.disabled = false;
  elements.stop.disabled = true;
  elements.hiddenLayers.disabled = false;
  elements.neurons.disabled = false;
  elements.status.textContent = message;
};

const trainingFrame = () => {
  if (!training) return;
  let currentLoss;
  for (let batch = 0; batch < 100; batch += 1) {
    currentLoss = network.trainEpoch(XOR_DATA);
    epoch += 1;
  }
  lossHistory.push(currentLoss);
  render();

  if (currentLoss < 0.0025 || epoch >= 50000) {
    finishTraining(
      currentLoss < 0.0025
        ? `Converged after ${epoch.toLocaleString()} epochs.`
        : "Stopped at the 50,000 epoch limit.",
    );
    return;
  }
  requestAnimationFrame(trainingFrame);
};

elements.train.addEventListener("click", () => {
  if (training) return;
  training = true;
  elements.train.disabled = true;
  elements.stop.disabled = false;
  elements.hiddenLayers.disabled = true;
  elements.neurons.disabled = true;
  elements.status.textContent = "Training…";
  requestAnimationFrame(trainingFrame);
});

elements.stop.addEventListener("click", () =>
  finishTraining("Training paused."),
);
elements.reset.addEventListener("click", () => {
  if (training) finishTraining("Training stopped and network reset.");
  buildNetwork();
});
elements.rate.addEventListener("input", () => {
  network.learningRate = Number(elements.rate.value);
  elements.rateOutput.textContent = Number(elements.rate.value).toFixed(2);
});
elements.hiddenLayers.addEventListener("change", buildNetwork);
elements.neurons.addEventListener("change", buildNetwork);
elements.export.addEventListener("click", async () => {
  const payload = JSON.stringify(
    { epoch, loss: network.loss(XOR_DATA), ...network.snapshot() },
    null,
    2,
  );
  try {
    await navigator.clipboard.writeText(payload);
    elements.status.textContent = "Model JSON copied to the clipboard.";
  } catch {
    elements.status.textContent =
      "Clipboard access was unavailable. Serve the page from localhost and try again.";
  }
});

buildNetwork();
