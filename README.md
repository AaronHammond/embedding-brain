# embedding-brain
A neural network library in Javascript, built to support input embeddings alongside ordinary inputs and output probabilities. Developed for personal explorations in machine learning, with general applicability in mind.

## Installation
	
	npm install embedding-brain

This library makes some use of features (only generators, currently) of ECMAScript 6. This means that your version of `node` must have support either for the `--harmony` execution flag or come out of the box with support for generators (most recent versions). Please see [the node.js documentation](https://nodejs.org/en/docs/es6/) for more information


## Usage

#### new NeuralNetwork(options)
Initialize a neural network with the corresponding options and random starting weights.

	options = {
		inputVectorSize: Number, // the number of non-embedding input signals to the network. default: 0
		embeddingSizes: [Number], // an array of integer sizes of each desired embedding vector. default: []
		hiddenLayerSizes: [Number], // an array of integer sizes (in node count) for each hidden layer. default [5] (one hidden layer with 5 nodes)
		outputSize: Number, // the number of output signals. default: 2 (for binary classification)
		learningRate: Number, // the learning rate for weight updates during training. default: 0.5
		inputLabels: [String], // string labels for each of the input signals. default: ["0", "1", ...]
		embeddingLabels: [String], // string labels for each input embedding vector. default: ["0", "1", ...]
		outputLabels: [String],  // string labels for each of the output signals. default: ["0", "1", ...]
	}

#### NeuralNetwork#train(labeledInputSignals, labeledEmbeddingVectors, labeledOutputSignals)
Train the neural network, with each input signal taking a value according to the labeled vector, each input embedding taking the vector matching its label, and each output signal taking a value according to the labeled vector. Returns an object of the form:

	{
		resultEmbeddings: {
			embeddingLabel: updatedEmbeddingVector
		}
	}

for each embedding label and its corresponding updated vector. See example.

#### NeuralNetwork#predict(labeledInputSignals, labeledEmbeddingVectors)
Compute (softmax) output signals with the neural network, with each input signal taking a value according to the labeled vector and each input embedding taking the vector matching its label. Returns an object of the form:

	{
		outputs: {
			outputLabel: outputSignal
		}
	}

for each output unit.

#### NeuralNetwork#reset()
Reset the weights learned in the network to random values.

#### Utils.Vector.random(n)
Return a vector (array) of n random values between 0 and 1 (exclusive). Useful for initializing embedding vectors before further training.

## Example
Run the example with `node example.js`.

	var Brain = require('embedding-brain');
	var Utils = Brain.Utils;

	// we are going to train a neural network to model the XOR function:
	// A | B | f(A, B)
	// 0   0   0
	// 0   1   1
    // 1   0   1
    // 1   1   0

    // to do this, we'll learn embeddings for each setting of A and B
    // each embedding will be a length 5 vector
	var nn = new Brain.NeuralNetwork({
		inputVectorSize: 0, // no non-embedding input signals
		embeddingSizes: [5], // one embedding input to the network with length 5
		hiddenLayerSizes: [3], // one hidden layer with 3 nodes
		outputSize: 2, // the output of the network is a softmax over 2 possible results, f(A, B) = 1 and f(A, B) = 0
		embeddingLabels: ["A:B"], // the embedding vector represents the input setting A,B for some A in [0, 1] and B in [0, 1]
		outputLabels: ["on", "off"], // the two output signals represent a probability for the output of the gate to be on or off
		learningRate: 0.8
	});

	// initialize random embedding vectors for each of the possible settings of A and B
	var em_0_0 = Utils.Vector.random(5);
	var em_0_1 = Utils.Vector.random(5);
	var em_1_0 = Utils.Vector.random(5);
	var em_1_1 = Utils.Vector.random(5);

	// train with our data for 100 iterations
	for (var i = 0; i < 100; i++) {
		// for each training example with a known result, provide 
		//   the labeled input signals (none),
		//   the labeled embedding vector for that training example (over settings of A,B)
		//   the labeled output signals for that training example (f(A, B))
		// and update the embedding for that training example's setting of A,B with the embedding after update
		em_0_0 = nn.train({}, {"A:B": em_0_0}, {"on": 0.0, "off": 1.0}).resultEmbeddings["A:B"];
		em_0_1 = nn.train({}, {"A:B": em_0_1}, {"on": 1.0, "off": 0.0}).resultEmbeddings["A:B"];
		em_1_0 = nn.train({}, {"A:B": em_1_0}, {"on": 1.0, "off": 0.0}).resultEmbeddings["A:B"];
		em_1_1 = nn.train({}, {"A:B": em_1_1}, {"on": 0.0, "off": 1.0}).resultEmbeddings["A:B"];
	}

	console.log(nn.predict({}, {"A:B": em_0_0})); // ex. { outputs: { "on": 0.006, "off": 0.993 }
	console.log(nn.predict({}, {"A:B": em_0_1})); // ex. { outputs: { "on": 0.996, "off": 0.003 }
	console.log(nn.predict({}, {"A:B": em_1_0})); // ex. { outputs: { "on": 0.997, "off": 0.002 }
	console.log(nn.predict({}, {"A:B": em_1_1})); // ex. { outputs: { "on": 0.005, "off": 0.994 }
