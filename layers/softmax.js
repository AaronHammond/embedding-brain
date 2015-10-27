var Vector = require('../utils/vector');
var Activation = require('../utils/activation');

var SoftMaxLayer = function (size, inputLayer, learningRate) {
	this.size = size;
	this.outputs = new Array(this.size);
	this.inputLayer = inputLayer;
	this.weights = new Array(this.size);
	this.learningRate = learningRate;

	for (var node_i = 0; node_i < this.size; node_i++) {
		this.weights[node_i] = Vector.random(this.inputLayer.size);
		this.weights[node_i].push(1);
	}

	this.forward = function () {
		this.inputLayer.forward();

		for (var node_i = 0; node_i < this.size; node_i++) {
			var z = 0;
			for (var in_i = 0; in_i < this.inputLayer.size; in_i++) {
				z += this.weights[node_i][in_i] * this.inputLayer.outputs[in_i];
			}
			this.outputs[node_i] = Math.exp(z);		
		}

		// normalize
		this.outputs = Vector.normalize(this.outputs);
	}

	this.target = function (outputs) {
		var deltas = new Array(this.size);
		for (var node_i = 0; node_i < this.size; node_i++) {
			deltas[node_i] = this.outputs[node_i] - outputs[node_i];
		}
		
		// partial deltas
		var weightDelta = Vector.zeros(this.inputLayer.size);

		for (var node_i = 0; node_i < this.size; node_i++) {
			for (var in_i = 0; in_i < this.inputLayer.size; in_i++) {
				// add to the update for that input
				weightDelta[in_i] += this.weights[node_i][in_i] * deltas[node_i];
				// update weight for the input
				this.weights[node_i][in_i] -= this.learningRate * deltas[node_i] * this.inputLayer.outputs[in_i];
			}

			this.weights[node_i][this.inputLayer.size] -= this.learningRate * deltas[node_i]; // bias
		}

		// compute deltas for input layer
		var outDeltas = new Array(this.inputLayer.size);
		for (var in_i = 0; in_i < outDeltas.length; in_i++) {
			outDeltas[in_i] = Activation.sigmoid.deriv(this.inputLayer.outputs[in_i]) * weightDelta[in_i];
		}

		this.inputLayer.backPropagate(outDeltas);
	}
}

module.exports = SoftMaxLayer;
