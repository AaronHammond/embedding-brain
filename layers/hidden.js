var Vector = require('../utils/vector');
var Activation = require('../utils/activation');

var HiddenLayer = function (size, inputLayer, learningRate) {
	this.size = size;
	this.outputs = new Array(this.size);
	this.inputLayer = inputLayer;
	this.weights = new Array(this.size);
	this.learningRate = learningRate;

	for (var node_i = 0; node_i < this.size; node_i++) {
		this.weights[node_i] = Vector.random(this.inputLayer.size);
		this.weights[node_i].push(1); // bias
	}
	this.forward = function () {
		var self = this;
		this.inputLayer.forward();
		for (var node_i = 0; node_i < this.size; node_i++) {
			this.outputs[node_i] = 0
			this.inputLayer.outputs.forEach(function (in_s, in_i) {
				self.outputs[node_i] += self.weights[node_i][in_i] * in_s;
			});
			this.outputs[node_i] += this.weights[node_i][this.inputLayer.size]; // bias
			this.outputs[node_i] = Activation.sigmoid.func(this.outputs[node_i]);
		}
		this.outputs[this.size]
	}

	this.backPropagate = function (deltas) {
		var weightDelta = Vector.zeros(this.inputLayer.size);

		for (var node_i = 0; node_i < this.size; node_i++) {
			for (var in_i = 0; in_i < this.inputLayer.size; in_i++) {
				weightDelta[in_i] += this.weights[node_i][in_i] * deltas[node_i];
				this.weights[node_i][in_i] -= this.learningRate * deltas[node_i] * this.inputLayer.outputs[in_i];
			}

			this.weights[node_i][this.inputLayer.size] -= this.learningRate * deltas[node_i]; // bias
		}

		var outDeltas = new Array(this.inputLayer.size);
		for (var in_i = 0; in_i < outDeltas.length; in_i++) {
			outDeltas[in_i] = Activation.sigmoid.deriv(this.inputLayer.outputs[in_i]) * weightDelta[in_i];
		}

		this.inputLayer.backPropagate(outDeltas);
	}
}

module.exports = HiddenLayer;
