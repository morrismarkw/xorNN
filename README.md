# Neural Network XOR Visualizer

A complete Salesforce application that visualizes a neural network learning the XOR function — built with a single Claude Code prompt.

## About

Every CS student has built this neural network in C. I just built it in Salesforce. Using a one-shot deployment prompt with Claude Code.

The XOR problem is a rite of passage — forward pass, backpropagation, sigmoid derivatives, weight updates. Raw loops, raw math. Everyone's done it in C or Python.

But Salesforce? With SVG visualizations, real-time training animations, SLDS styling, and automatic activity logging via Lightning Data Service?

That's a different exercise entirely.

I gave Claude Code a single prompt. It:
- Scaffolded the SFDX project
- Generated the LWC (600+ lines of clean, readable JS — no ML libraries)
- Created the custom object, fields, list views, tabs, permission sets
- Deployed everything to my org
- Assigned permissions and opened the app

Zero to deployed. One prompt.

The neural network code reads exactly like you'd write it in C — raw for loops, explicit gradients, no abstraction layers. Except now it runs in a browser, animates in real-time, and logs every training run to a Salesforce object.

Sometimes the best way to understand something is to build it somewhere it was never meant to exist.

## Features

- **Configurable Network Topology**: 1-4 hidden layers, 1-8 neurons per layer
- **Real-time SVG Visualization**: Watch weights and activations animate during training
- **Live Loss Chart**: See the loss curve update each epoch
- **XOR Truth Table**: Live predictions vs expected outputs
- **Activity Logging**: Every action logged to NN_Log__c via Lightning Data Service
- **Export Results**: Copy trained weights/biases to clipboard as JSON
- **Full SLDS Styling**: Professional Salesforce Lightning look and feel

## Deployment

```bash
sf org login web --set-default
sf project deploy start --source-dir force-app
sf org assign permset --name Neural_Network_User
sf org open --path "/lightning/app/c__Neural_Network"
```

---

# The One-Shot Deployment Prompt

The following prompt was used to generate this entire application:

---

Build a complete, deployable Salesforce application that visualizes a neural network learning the XOR function.

PREREQUISITES:
- Verify the Salesforce CLI (sf) is installed; if not, provide installation instructions
- Create a new SFDX project in the current directory if one doesn't exist
- Ensure a default org is connected; if not, authenticate to a Production/Developer Edition org and set it as the default
- After deployment, assign the permission set to the current user and open the app

SALESFORCE APP & METADATA:
- Lightning App called "Neural Network" (API name: Neural_Network) using the Cosmos theme
- Custom Tab hosting the main LWC component, added to the Neural Network app
- Permission Set called "Neural Network User" granting access to the app, tab, and all custom objects

CUSTOM OBJECT — NN_Log__c:
- Custom object called "NN Log" (API name: NN_Log__c) to track user activity in the app
- Fields: User__c (Lookup to User), Action__c (Text — e.g. "Training Started", "Training Completed", "Learning Rate Changed", "Reset"), Epochs__c (Number — epoch count at time of action), Final_Loss__c (Number — loss value at time of action), Learning_Rate__c (Number — current learning rate), Timestamp__c (DateTime)
- Create an "All" list view showing all important fields (Name, User, Action, Epochs, Final_Loss, Learning_Rate, Timestamp, Created Date), available to all internal users
- Use Lightning Data Service via lightning/uiRecordApi createRecord to write NN_Log__c records from the component — no Apex classes

CONFIGURABLE NETWORK ARCHITECTURE:
- The network topology must be fully configurable: number of inputs, number of hidden layers, number of neurons per hidden layer, and number of outputs
- Defaults: 2 inputs, 1 hidden layer with 2 neurons, 1 output (classic XOR configuration)
- UI controls to adjust topology before training — input fields or dropdowns for: number of hidden layers (1-4), neurons per layer (1-8), learning rate (slider 0.01 to 10.0)
- When topology changes, the SVG visualization and all internal data structures rebuild dynamically
- The goal is to show how simple these algorithms actually are — the forward pass, backpropagation, and weight updates should be written as clean, readable, well-commented JavaScript that mirrors how you'd write it in C. No abstraction layers or ML library patterns. Raw loops, raw math. Every function should be obvious: sigmoid, sigmoidDerivative, forwardPass, backpropagate, updateWeights.

VISUALIZATION:
- Render the full network diagram using SVG within the LWC template — scales dynamically with topology
- Connection weights shown as line thickness and color (green for positive, red for negative)
- Nodes display their current activation value during and after training
- Network animates during training as weights and activations update in real time

TRAINED MODEL RESULTS PANEL:
- After training completes, display a detailed results panel showing:
  - All final weights organized by layer and connection (e.g. "Layer 1, Neuron 1, Input 2: 5.734")
  - All final biases per neuron
  - Final activation values for each XOR input combination run through the trained network
  - Total epochs run, final loss value, and learning rate used
  - Format this as a clean SLDS data table so it's easy to read and screenshot
- Include an "Export Results" button that copies the full weights/biases to clipboard as JSON

TRAINING UI — use SLDS classes, spacing tokens, and component blueprints throughout:
- "Train" button that runs backpropagation epochs in real time
- "Reset" button to reinitialize random weights and clear training
- Epoch counter display
- Loss chart using SVG that updates each epoch showing the loss curve
- Slider to adjust learning rate
- XOR truth table showing current predictions vs expected outputs, updating live during training
- Every significant user action (train start, train complete, rate change, topology change, reset) creates an NN_Log__c record via LDS

TECHNICAL CONSTRAINTS:
- LWC reactive framework with tracked properties for all dynamic values
- All rendering in HTML template with SVG — no D3 or external libraries
- Training loop uses requestAnimationFrame or setTimeout so the UI updates between epochs rather than freezing
- Generate every deployable file: LWC bundle (JS, HTML, CSS, js-meta.xml), custom object and field XML, list view XML, permission set XML, Lightning app XML, and custom tab XML — all in correct sfdx source format directory structure
- Deploy all components to the connected org using sf project deploy start

---

## License

MIT

## Built With

- [Claude Code](https://claude.ai/claude-code) — AI-powered coding assistant
- [Salesforce Lightning Web Components](https://developer.salesforce.com/docs/component-library/documentation/en/lwc)
