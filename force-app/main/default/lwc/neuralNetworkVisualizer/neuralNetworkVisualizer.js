/**
 * Neural Network XOR Visualizer
 *
 * A complete, from-scratch implementation of a feedforward neural network
 * that learns the XOR function. No ML libraries - just raw math and loops.
 *
 * The goal is to show how simple these algorithms actually are.
 */
import { LightningElement, track } from 'lwc';
import { createRecord } from 'lightning/uiRecordApi';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';
import Id from '@salesforce/user/Id';

import NN_LOG_OBJECT from '@salesforce/schema/NN_Log__c';
import USER_FIELD from '@salesforce/schema/NN_Log__c.User__c';
import ACTION_FIELD from '@salesforce/schema/NN_Log__c.Action__c';
import EPOCHS_FIELD from '@salesforce/schema/NN_Log__c.Epochs__c';
import FINAL_LOSS_FIELD from '@salesforce/schema/NN_Log__c.Final_Loss__c';
import LEARNING_RATE_FIELD from '@salesforce/schema/NN_Log__c.Learning_Rate__c';
import TIMESTAMP_FIELD from '@salesforce/schema/NN_Log__c.Timestamp__c';

export default class NeuralNetworkVisualizer extends LightningElement {
    // =========================================================================
    // NETWORK CONFIGURATION - User can adjust these before training
    // =========================================================================
    @track numInputs = 2;           // XOR has 2 inputs
    @track numOutputs = 1;          // XOR has 1 output
    @track numHiddenLayers = 1;     // Number of hidden layers (1-4)
    @track neuronsPerLayer = 2;     // Neurons in each hidden layer (1-8)
    @track learningRate = 2.0;      // Step size for gradient descent

    // =========================================================================
    // NETWORK STATE - The actual weights, biases, and activations
    // =========================================================================
    @track weights = [];    // weights[layer][toNeuron][fromNeuron]
    @track biases = [];     // biases[layer][neuron]
    @track activations = []; // activations[layer][neuron] - output of each neuron
    @track preActivations = []; // z values before sigmoid

    // =========================================================================
    // TRAINING STATE
    // =========================================================================
    @track isTraining = false;
    @track epoch = 0;
    @track currentLoss = 0;
    @track lossHistory = [];
    @track trainingComplete = false;
    animationFrameId = null;

    // XOR training data - the four possible input combinations
    trainingData = [
        { inputs: [0, 0], expected: [0] },
        { inputs: [0, 1], expected: [1] },
        { inputs: [1, 0], expected: [1] },
        { inputs: [1, 1], expected: [0] }
    ];

    // =========================================================================
    // UI STATE - For displaying predictions and results
    // =========================================================================
    @track predictions = [
        { inputs: '0, 0', expected: 0, predicted: 0, correct: false },
        { inputs: '0, 1', expected: 1, predicted: 0, correct: false },
        { inputs: '1, 0', expected: 1, predicted: 0, correct: false },
        { inputs: '1, 1', expected: 0, predicted: 0, correct: false }
    ];

    @track showResults = false;
    @track finalWeightsData = [];
    @track finalBiasesData = [];
    @track finalActivationsData = [];

    // SVG dimensions
    svgWidth = 800;
    svgHeight = 400;
    lossChartWidth = 400;
    lossChartHeight = 150;

    // Current user ID for logging
    userId = Id;

    // =========================================================================
    // LIFECYCLE
    // =========================================================================
    connectedCallback() {
        this.initializeNetwork();
    }

    // =========================================================================
    // CONFIGURATION OPTIONS FOR UI
    // =========================================================================
    get hiddenLayerOptions() {
        return [
            { label: '1 Hidden Layer', value: '1' },
            { label: '2 Hidden Layers', value: '2' },
            { label: '3 Hidden Layers', value: '3' },
            { label: '4 Hidden Layers', value: '4' }
        ];
    }

    get neuronsPerLayerOptions() {
        return [
            { label: '1 Neuron', value: '1' },
            { label: '2 Neurons', value: '2' },
            { label: '3 Neurons', value: '3' },
            { label: '4 Neurons', value: '4' },
            { label: '5 Neurons', value: '5' },
            { label: '6 Neurons', value: '6' },
            { label: '7 Neurons', value: '7' },
            { label: '8 Neurons', value: '8' }
        ];
    }

    get selectedHiddenLayers() {
        return String(this.numHiddenLayers);
    }

    get selectedNeuronsPerLayer() {
        return String(this.neuronsPerLayer);
    }

    get learningRateDisplay() {
        return this.learningRate.toFixed(2);
    }

    get epochDisplay() {
        return this.epoch.toLocaleString();
    }

    get lossDisplay() {
        return this.currentLoss.toFixed(6);
    }

    get canTrain() {
        return !this.isTraining;
    }

    get trainButtonLabel() {
        return this.isTraining ? 'Training...' : 'Train Network';
    }

    get trainButtonVariant() {
        return this.isTraining ? 'neutral' : 'brand';
    }

    // =========================================================================
    // ACTIVATION FUNCTION: SIGMOID
    //
    // The sigmoid function squashes any input to a value between 0 and 1.
    // Formula: σ(x) = 1 / (1 + e^(-x))
    //
    // This is what gives neural networks their non-linearity.
    // Without non-linear activation, stacking layers would be pointless -
    // multiple linear transformations just collapse into one.
    // =========================================================================
    sigmoid(x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // =========================================================================
    // SIGMOID DERIVATIVE
    //
    // The derivative of sigmoid is elegantly simple: σ'(x) = σ(x) * (1 - σ(x))
    // We need this for backpropagation to compute gradients.
    //
    // Note: We pass in the already-computed sigmoid value, not the raw x,
    // since we already have σ(x) stored in our activations.
    // =========================================================================
    sigmoidDerivative(sigmoidOutput) {
        return sigmoidOutput * (1.0 - sigmoidOutput);
    }

    // =========================================================================
    // NETWORK INITIALIZATION
    //
    // Set up the network structure based on current configuration.
    // Initialize all weights to small random values (Xavier-ish initialization).
    // Initialize all biases to zero.
    // =========================================================================
    initializeNetwork() {
        // Build the layer sizes array
        // [numInputs, neuronsPerLayer, neuronsPerLayer, ..., numOutputs]
        const layerSizes = [this.numInputs];
        for (let i = 0; i < this.numHiddenLayers; i++) {
            layerSizes.push(this.neuronsPerLayer);
        }
        layerSizes.push(this.numOutputs);

        // Initialize weights and biases
        // weights[l][j][i] = weight from neuron i in layer l to neuron j in layer l+1
        this.weights = [];
        this.biases = [];
        this.activations = [];
        this.preActivations = [];

        for (let l = 0; l < layerSizes.length - 1; l++) {
            const fromSize = layerSizes[l];
            const toSize = layerSizes[l + 1];

            // Initialize weights with small random values
            // Using a simple uniform distribution scaled by sqrt(2/fan_in)
            const scale = Math.sqrt(2.0 / fromSize);
            const layerWeights = [];
            for (let j = 0; j < toSize; j++) {
                const neuronWeights = [];
                for (let i = 0; i < fromSize; i++) {
                    // Random value between -scale and +scale
                    neuronWeights.push((Math.random() * 2 - 1) * scale);
                }
                layerWeights.push(neuronWeights);
            }
            this.weights.push(layerWeights);

            // Initialize biases to zero
            const layerBiases = [];
            for (let j = 0; j < toSize; j++) {
                layerBiases.push(0);
            }
            this.biases.push(layerBiases);
        }

        // Initialize activations array (will be populated during forward pass)
        for (let l = 0; l < layerSizes.length; l++) {
            this.activations.push(new Array(layerSizes[l]).fill(0));
            this.preActivations.push(new Array(layerSizes[l]).fill(0));
        }

        // Reset training state
        this.epoch = 0;
        this.currentLoss = 0;
        this.lossHistory = [];
        this.trainingComplete = false;
        this.showResults = false;
        this.updatePredictions();
    }

    // =========================================================================
    // FORWARD PASS
    //
    // Feed an input through the network and compute the output.
    // This is the "inference" step - just matrix multiplication + activation.
    //
    // For each layer:
    //   z[j] = sum(w[j][i] * a[i]) + b[j]   (weighted sum + bias)
    //   a[j] = sigmoid(z[j])                 (apply activation)
    //
    // Returns the output layer activations.
    // =========================================================================
    forwardPass(inputs) {
        // Set input layer activations to the input values
        for (let i = 0; i < inputs.length; i++) {
            this.activations[0][i] = inputs[i];
            this.preActivations[0][i] = inputs[i];
        }

        // Propagate forward through each layer
        for (let l = 0; l < this.weights.length; l++) {
            const fromActivations = this.activations[l];
            const toSize = this.weights[l].length;

            for (let j = 0; j < toSize; j++) {
                // Compute weighted sum: z = Σ(w[j][i] * a[i]) + b[j]
                let z = this.biases[l][j];
                for (let i = 0; i < fromActivations.length; i++) {
                    z += this.weights[l][j][i] * fromActivations[i];
                }

                // Store pre-activation value (needed for backprop)
                this.preActivations[l + 1][j] = z;

                // Apply sigmoid activation
                this.activations[l + 1][j] = this.sigmoid(z);
            }
        }

        // Return output layer activations
        return this.activations[this.activations.length - 1];
    }

    // =========================================================================
    // BACKPROPAGATION
    //
    // The heart of neural network learning. We compute how much each weight
    // contributed to the error, then adjust weights to reduce that error.
    //
    // The algorithm works backwards from the output:
    // 1. Compute output error: δ[output] = (predicted - expected) * σ'(z)
    // 2. Propagate error back: δ[l] = (W[l+1]^T · δ[l+1]) * σ'(z[l])
    // 3. Compute gradients: ∂Loss/∂w[l][j][i] = δ[l+1][j] * a[l][i]
    //
    // This is just the chain rule applied systematically.
    // =========================================================================
    backpropagate(expected) {
        const numLayers = this.activations.length;

        // Initialize delta (error) arrays for each layer
        const deltas = [];
        for (let l = 0; l < numLayers; l++) {
            deltas.push(new Array(this.activations[l].length).fill(0));
        }

        // Step 1: Compute output layer deltas
        // δ[j] = (a[j] - expected[j]) * σ'(a[j])
        const outputLayer = numLayers - 1;
        for (let j = 0; j < this.activations[outputLayer].length; j++) {
            const output = this.activations[outputLayer][j];
            const error = output - expected[j];
            deltas[outputLayer][j] = error * this.sigmoidDerivative(output);
        }

        // Step 2: Backpropagate deltas through hidden layers
        // δ[l][i] = (Σ w[l][j][i] * δ[l+1][j]) * σ'(a[l][i])
        for (let l = numLayers - 2; l > 0; l--) {
            for (let i = 0; i < this.activations[l].length; i++) {
                let errorSum = 0;
                // Sum contributions from all neurons this one connects to
                for (let j = 0; j < this.activations[l + 1].length; j++) {
                    errorSum += this.weights[l][j][i] * deltas[l + 1][j];
                }
                deltas[l][i] = errorSum * this.sigmoidDerivative(this.activations[l][i]);
            }
        }

        return deltas;
    }

    // =========================================================================
    // UPDATE WEIGHTS
    //
    // Apply gradient descent to adjust weights and biases.
    //
    // w[l][j][i] -= learningRate * δ[l+1][j] * a[l][i]
    // b[l][j] -= learningRate * δ[l+1][j]
    //
    // The learning rate controls how big of a step we take.
    // Too small = slow learning. Too big = overshooting and instability.
    // =========================================================================
    updateWeights(deltas) {
        for (let l = 0; l < this.weights.length; l++) {
            for (let j = 0; j < this.weights[l].length; j++) {
                // Update each weight connecting layer l to layer l+1
                for (let i = 0; i < this.weights[l][j].length; i++) {
                    const gradient = deltas[l + 1][j] * this.activations[l][i];
                    this.weights[l][j][i] -= this.learningRate * gradient;
                }
                // Update bias
                this.biases[l][j] -= this.learningRate * deltas[l + 1][j];
            }
        }
    }

    // =========================================================================
    // COMPUTE LOSS (Mean Squared Error)
    //
    // Loss = (1/n) * Σ(predicted - expected)²
    //
    // This measures how wrong the network is. We want to minimize this.
    // =========================================================================
    computeLoss() {
        let totalLoss = 0;

        for (let sample of this.trainingData) {
            const outputs = this.forwardPass(sample.inputs);
            for (let i = 0; i < outputs.length; i++) {
                const error = outputs[i] - sample.expected[i];
                totalLoss += error * error;
            }
        }

        return totalLoss / this.trainingData.length;
    }

    // =========================================================================
    // TRAINING STEP
    //
    // One complete pass through all training samples.
    // For each sample: forward pass → compute error → backprop → update weights
    // =========================================================================
    trainOneEpoch() {
        // Train on each sample in the training set
        for (let sample of this.trainingData) {
            // Forward pass - compute network output
            this.forwardPass(sample.inputs);

            // Backward pass - compute gradients
            const deltas = this.backpropagate(sample.expected);

            // Update weights using gradients
            this.updateWeights(deltas);
        }

        this.epoch++;
        this.currentLoss = this.computeLoss();

        // Store loss history for the chart
        if (this.epoch % 10 === 0 || this.epoch <= 50) {
            this.lossHistory = [...this.lossHistory, { epoch: this.epoch, loss: this.currentLoss }];
        }

        // Update predictions display
        this.updatePredictions();
    }

    // =========================================================================
    // UPDATE PREDICTIONS TABLE
    //
    // Run each XOR input through the network and update the display.
    // =========================================================================
    updatePredictions() {
        const newPredictions = [];

        for (let sample of this.trainingData) {
            const outputs = this.forwardPass(sample.inputs);
            const predicted = outputs[0];
            const rounded = Math.round(predicted);

            newPredictions.push({
                inputs: sample.inputs.join(', '),
                expected: sample.expected[0],
                predicted: predicted.toFixed(4),
                predictedRounded: rounded,
                correct: rounded === sample.expected[0]
            });
        }

        this.predictions = newPredictions;
    }

    // =========================================================================
    // TRAINING LOOP
    //
    // Run training epochs with animation frames so UI can update.
    // Uses requestAnimationFrame for smooth rendering.
    // =========================================================================
    runTrainingLoop() {
        if (!this.isTraining) return;

        // Run multiple epochs per frame for faster training
        const epochsPerFrame = 5;
        for (let i = 0; i < epochsPerFrame; i++) {
            this.trainOneEpoch();
        }

        // Check convergence - stop if loss is very low
        if (this.currentLoss < 0.0001) {
            this.stopTraining();
            this.onTrainingComplete();
            return;
        }

        // Check max epochs
        if (this.epoch >= 50000) {
            this.stopTraining();
            this.onTrainingComplete();
            return;
        }

        // Schedule next frame
        this.animationFrameId = requestAnimationFrame(() => this.runTrainingLoop());
    }

    // =========================================================================
    // UI EVENT HANDLERS
    // =========================================================================

    handleHiddenLayersChange(event) {
        const oldValue = this.numHiddenLayers;
        this.numHiddenLayers = parseInt(event.detail.value, 10);
        this.initializeNetwork();

        if (oldValue !== this.numHiddenLayers) {
            this.logAction('Topology Changed', `Hidden layers: ${this.numHiddenLayers}`);
        }
    }

    handleNeuronsChange(event) {
        const oldValue = this.neuronsPerLayer;
        this.neuronsPerLayer = parseInt(event.detail.value, 10);
        this.initializeNetwork();

        if (oldValue !== this.neuronsPerLayer) {
            this.logAction('Topology Changed', `Neurons per layer: ${this.neuronsPerLayer}`);
        }
    }

    handleLearningRateChange(event) {
        this.learningRate = parseFloat(event.target.value);
    }

    handleLearningRateCommit() {
        this.logAction('Learning Rate Changed');
    }

    handleTrain() {
        if (this.isTraining) return;

        this.isTraining = true;
        this.trainingComplete = false;
        this.showResults = false;
        this.logAction('Training Started');

        // Start the training loop
        this.runTrainingLoop();
    }

    handleStop() {
        this.stopTraining();
        this.logAction('Training Stopped');
    }

    stopTraining() {
        this.isTraining = false;
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    handleReset() {
        this.stopTraining();
        this.initializeNetwork();
        this.logAction('Reset');
    }

    onTrainingComplete() {
        this.trainingComplete = true;
        this.showResults = true;
        this.generateResultsData();
        this.logAction('Training Completed');
    }

    // =========================================================================
    // RESULTS GENERATION
    // =========================================================================

    generateResultsData() {
        // Generate weights table data
        const weightsData = [];
        for (let l = 0; l < this.weights.length; l++) {
            for (let j = 0; j < this.weights[l].length; j++) {
                for (let i = 0; i < this.weights[l][j].length; i++) {
                    weightsData.push({
                        id: `w-${l}-${j}-${i}`,
                        layer: l + 1,
                        toNeuron: j + 1,
                        fromNeuron: i + 1,
                        value: this.weights[l][j][i].toFixed(6),
                        description: `Layer ${l + 1} → ${l + 2}, Neuron ${i + 1} → ${j + 1}`
                    });
                }
            }
        }
        this.finalWeightsData = weightsData;

        // Generate biases table data
        const biasesData = [];
        for (let l = 0; l < this.biases.length; l++) {
            for (let j = 0; j < this.biases[l].length; j++) {
                biasesData.push({
                    id: `b-${l}-${j}`,
                    layer: l + 2,
                    neuron: j + 1,
                    value: this.biases[l][j].toFixed(6)
                });
            }
        }
        this.finalBiasesData = biasesData;

        // Generate final activations for each XOR input
        const activationsData = [];
        for (let sample of this.trainingData) {
            const outputs = this.forwardPass(sample.inputs);
            activationsData.push({
                id: `act-${sample.inputs.join('-')}`,
                inputs: sample.inputs.join(', '),
                expected: sample.expected[0],
                output: outputs[0].toFixed(6),
                rounded: Math.round(outputs[0])
            });
        }
        this.finalActivationsData = activationsData;
    }

    handleExportResults() {
        const exportData = {
            configuration: {
                numInputs: this.numInputs,
                numHiddenLayers: this.numHiddenLayers,
                neuronsPerLayer: this.neuronsPerLayer,
                numOutputs: this.numOutputs,
                learningRate: this.learningRate
            },
            training: {
                epochs: this.epoch,
                finalLoss: this.currentLoss
            },
            weights: this.weights,
            biases: this.biases,
            predictions: this.finalActivationsData
        };

        const jsonString = JSON.stringify(exportData, null, 2);

        // Copy to clipboard
        navigator.clipboard.writeText(jsonString).then(() => {
            this.dispatchEvent(
                new ShowToastEvent({
                    title: 'Exported',
                    message: 'Network weights and configuration copied to clipboard',
                    variant: 'success'
                })
            );
        }).catch(() => {
            this.dispatchEvent(
                new ShowToastEvent({
                    title: 'Export Failed',
                    message: 'Could not copy to clipboard',
                    variant: 'error'
                })
            );
        });
    }

    // =========================================================================
    // LOGGING - Create NN_Log__c records via Lightning Data Service
    // =========================================================================

    logAction(action, detail = '') {
        const fields = {};
        fields[USER_FIELD.fieldApiName] = this.userId;
        fields[ACTION_FIELD.fieldApiName] = detail ? `${action}: ${detail}` : action;
        fields[EPOCHS_FIELD.fieldApiName] = this.epoch;
        fields[FINAL_LOSS_FIELD.fieldApiName] = this.currentLoss;
        fields[LEARNING_RATE_FIELD.fieldApiName] = this.learningRate;
        fields[TIMESTAMP_FIELD.fieldApiName] = new Date().toISOString();

        const recordInput = { apiName: NN_LOG_OBJECT.objectApiName, fields };

        createRecord(recordInput)
            .then(() => {
                // Record created successfully
            })
            .catch(error => {
                console.error('Error creating log record:', error);
            });
    }

    // =========================================================================
    // SVG VISUALIZATION - Network Diagram
    // =========================================================================

    get networkVisualization() {
        const layers = this.getLayerSizes();
        const nodes = [];
        const connections = [];

        const padding = 60;
        const layerSpacing = (this.svgWidth - 2 * padding) / (layers.length - 1);
        const maxNeurons = Math.max(...layers);

        // Create nodes for each layer
        const nodePositions = [];
        for (let l = 0; l < layers.length; l++) {
            const x = padding + l * layerSpacing;
            const neuronCount = layers[l];
            const neuronSpacing = (this.svgHeight - 2 * padding) / (neuronCount + 1);

            nodePositions.push([]);
            for (let n = 0; n < neuronCount; n++) {
                const y = padding + (n + 1) * neuronSpacing;
                nodePositions[l].push({ x, y });

                const activation = this.activations[l] ? this.activations[l][n] : 0;
                const intensity = Math.floor(activation * 200);
                const fillColor = `rgb(${55 + intensity}, ${100 + intensity}, ${200})`;

                // Get bias for this neuron (input layer has no bias)
                const hasBias = l > 0;
                const biasIndex = l - 1; // biases array is indexed from 0 for first hidden layer
                const bias = hasBias && this.biases[biasIndex] ? this.biases[biasIndex][n] : 0;

                nodes.push({
                    id: `node-${l}-${n}`,
                    cx: x,
                    cy: y,
                    r: 22,
                    fill: fillColor,
                    stroke: l === 0 ? '#1589ee' : (l === layers.length - 1 ? '#4bca81' : '#706e6b'),
                    strokeWidth: 3,
                    label: activation.toFixed(2),
                    labelX: x,
                    labelY: y + 5,
                    layerLabel: l === 0 ? 'Input' : (l === layers.length - 1 ? 'Output' : `Hidden ${l}`),
                    hasBias: hasBias,
                    biasLabel: hasBias ? `b: ${bias.toFixed(2)}` : '',
                    biasX: x,
                    biasY: y + 35,
                    biasColor: bias >= 0 ? '#4bca81' : '#ea5252'
                });
            }
        }

        // Create connections between layers
        for (let l = 0; l < layers.length - 1; l++) {
            for (let from = 0; from < layers[l]; from++) {
                for (let to = 0; to < layers[l + 1]; to++) {
                    const weight = this.weights[l] ? this.weights[l][to][from] : 0;
                    const absWeight = Math.abs(weight);
                    const strokeWidth = Math.min(Math.max(absWeight * 2, 0.5), 6);
                    const strokeColor = weight >= 0 ?
                        `rgba(75, 202, 129, ${Math.min(absWeight / 3, 0.9)})` :
                        `rgba(234, 82, 82, ${Math.min(absWeight / 3, 0.9)})`;

                    connections.push({
                        id: `conn-${l}-${from}-${to}`,
                        x1: nodePositions[l][from].x,
                        y1: nodePositions[l][from].y,
                        x2: nodePositions[l + 1][to].x,
                        y2: nodePositions[l + 1][to].y,
                        strokeWidth: strokeWidth,
                        stroke: strokeColor
                    });
                }
            }
        }

        return { nodes, connections };
    }

    get layerLabels() {
        const layers = this.getLayerSizes();
        const labels = [];
        const padding = 60;
        const layerSpacing = (this.svgWidth - 2 * padding) / (layers.length - 1);

        for (let l = 0; l < layers.length; l++) {
            const x = padding + l * layerSpacing;
            let label = '';
            if (l === 0) {
                label = 'Input';
            } else if (l === layers.length - 1) {
                label = 'Output';
            } else {
                label = `Hidden ${l}`;
            }
            labels.push({ id: `label-${l}`, x, y: 25, text: label });
        }

        return labels;
    }

    getLayerSizes() {
        const sizes = [this.numInputs];
        for (let i = 0; i < this.numHiddenLayers; i++) {
            sizes.push(this.neuronsPerLayer);
        }
        sizes.push(this.numOutputs);
        return sizes;
    }

    // =========================================================================
    // SVG VISUALIZATION - Loss Chart
    // =========================================================================

    get lossChartPath() {
        if (this.lossHistory.length < 2) {
            return '';
        }

        const padding = 40;
        const width = this.lossChartWidth - 2 * padding;
        const height = this.lossChartHeight - 2 * padding;

        const maxEpoch = Math.max(...this.lossHistory.map(p => p.epoch));
        const maxLoss = Math.max(...this.lossHistory.map(p => p.loss), 0.5);

        const points = this.lossHistory.map(point => {
            const x = padding + (point.epoch / maxEpoch) * width;
            const y = padding + height - (point.loss / maxLoss) * height;
            return `${x},${y}`;
        });

        return `M ${points.join(' L ')}`;
    }

    get lossChartAxes() {
        const padding = 40;
        const width = this.lossChartWidth - 2 * padding;
        const height = this.lossChartHeight - 2 * padding;

        return {
            xAxis: {
                x1: padding,
                y1: padding + height,
                x2: padding + width,
                y2: padding + height
            },
            yAxis: {
                x1: padding,
                y1: padding,
                x2: padding,
                y2: padding + height
            },
            xLabel: { x: padding + width / 2, y: this.lossChartHeight - 5, text: 'Epoch' },
            yLabel: { x: 15, y: padding + height / 2, text: 'Loss' }
        };
    }

    get hasLossData() {
        return this.lossHistory.length > 1;
    }

    // =========================================================================
    // SUMMARY STATS
    // =========================================================================

    get allPredictionsCorrect() {
        return this.predictions.every(p => p.correct);
    }

    get accuracyPercentage() {
        const correct = this.predictions.filter(p => p.correct).length;
        return ((correct / this.predictions.length) * 100).toFixed(0);
    }
}
