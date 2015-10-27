// sizes defines the sizes of embeddings
// input_size defines the number of ordinary signals
// outputs : [embedding_0[0], embedding_0[1], ... , embedding_1[0], ...., embedding_n[k], signals[0], signals[1], ... , signals[j]]
var Vector = require('../utils/vector');
var Activation = require('../utils/activation');

var EmbeddingLayer = function (sizes, inputSize) {
	var self = this;
	this.size = sizes.reduce(function (prev, curr) { return prev + curr }, 0) + inputSize;
	this.outputs = new Array(this.size);
	// randomly init embeddings
	this.embeddings = new Array(sizes.length);
	sizes.forEach(function (size, emb_i) {
		self.embeddings[emb_i] = Vector.random(size);
	});
	// randomly init signals
	this.signals = Vector.random(inputSize);

	this.setEmbeddings = function (embeddingVectors) {
		this.embeddings = embeddingVectors;
	}

	this.setSignals = function (signals) {
		this.signals = signals;
	}

	this.backPropagate = function (deltas) {
		// only update the embeddings
		var delta = (function*(arr) { yield* arr; })(deltas.slice(0, this.size - this.signals.length));

		for (var emb_i = 0; emb_i < this.embeddings.length; emb_i++) {
			for (var v_i = 0; v_i < this.embeddings[emb_i].length; v_i++) {
				this.embeddings[emb_i][v_i] -= delta.next().value;
			}
		}
	}

	this.forward = function () {
		var extractOutputs = (function* (arrs) {
			for (var i = 0; i < arrs.length; i++) {
				for (var j = 0; j < arrs[i].length; j++) {
					yield arrs[i][j]
				}
			}
		})(this.embeddings);

		for (var v_i = 0; v_i < this.size - this.signals.length; v_i++) {
			this.outputs[v_i] = extractOutputs.next().value;
		}

		for (var v_o = 0; v_o < this.signals.length; v_o++) {
			this.outputs[this.embeddings.length + v_o] = this.signals[v_o];
		}
	}
}

module.exports = EmbeddingLayer;
