# JavaScript XOR neural network

This package contains a reusable feedforward neural network and an interactive browser visualizer. It uses no runtime or test dependencies.

The default topology is 2–2–1. Training uses the four XOR examples, sigmoid activation, mean squared error, and online backpropagation.

## Run

From the repository root:

```bash
python3 -m http.server 8000
```

Open <http://localhost:8000/javascript/>.

## Test

```bash
npm test
```

The tests cover deterministic forward propagation, parameter validation, model shape, and seeded XOR convergence.
