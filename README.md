# xorNN

Two from-scratch implementations of the same small neural-network exercise: teach a feedforward network to reproduce the XOR truth table with sigmoid activations and backpropagation.

| Implementation               | Runtime              | What it demonstrates                                                          |
| ---------------------------- | -------------------- | ----------------------------------------------------------------------------- |
| [`javascript/`](javascript/) | Any modern browser   | A dependency-free interactive page with a reusable, tested JavaScript network |
| [`salesforce/`](salesforce/) | Salesforce Lightning | A configurable LWC visualizer plus Salesforce metadata and activity logging   |

Both implementations use explicit arrays, loops, forward propagation, and gradient descent rather than a machine-learning library. They are independent implementations of the same idea; their serialized weights are not interchangeable.

## JavaScript

Run the tests:

```bash
npm test
```

Open the interactive visualizer from a local web server:

```bash
python3 -m http.server 8000
```

Then visit <http://localhost:8000/javascript/>.

## Salesforce

The complete Salesforce DX project and its own documentation live in [`salesforce/`](salesforce/). Run Salesforce package commands from that directory:

```bash
cd salesforce
npm install
npm run lint
npm test
```

Deployments are intentionally not part of the root scripts. Review the target org and the instructions in [`salesforce/README.md`](salesforce/README.md) before running Salesforce CLI commands.

## History

This repository began as the Salesforce implementation. The standalone JavaScript version consolidates the useful neural-network behavior from the former private `nn` repository into a clean browser application. Generated environment exports and the empty `neuralnetwork` repository were intentionally not carried forward.

## License

MIT
