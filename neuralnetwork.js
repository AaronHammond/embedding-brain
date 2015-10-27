var Vector = require('./utils/vector');
var HiddenLayer = require('./layers/hidden');
var EmbeddingLayer = require('./layers/embedding');
var SoftMaxLayer = require('./layers/softmax');

var NeuralNetwork = function (opts) {
	// BEGIN options
	this.inputVectorSize = opts.inputVectorSize || 0;
	
	this.embeddingSizes = opts.embeddingSizes || [];
	this.embeddingCount = this.embeddingSizes.length;
	
	this.hiddenLayerSizes = opts.hiddenLayerSizes || [5];
	this.hiddenLayerCount = this.hiddenLayerSizes.length;

	this.outputSize = opts.outputSize || 2;

	this.learningRate = opts.learningRate || 0.5; 

	function labeler (size, labels) { 
		var labeled = {}; 
		labels = (labels || []);
		for (var i = 0; i < size; i++) { 
			var l = (labels[i] || String(i));
			labeled[l] = i;
		} 
		return labeled; 
	}
	this.inputLabels = labeler(this.inputVectorSize, opts.inputLabels);
	this.embeddingLabels = labeler(this.embeddingSizes.length, opts.embeddingLabels);
	this.outputLabels = labeler(this.outputSize, opts.outputLabels);
	this.hiddenLayers = new Array(this.hiddenLayerSizes.length);
	// END options

	this.reset = function () {
		this.inputLayer = new EmbeddingLayer(this.embeddingSizes, this.inputVectorSize, this.learningRate);
		var prevLayer = this.inputLayer;

		for (var i = 0; i < this.hiddenLayerCount; i++) {
			this.hiddenLayers[i] = new HiddenLayer(this.hiddenLayerSizes[i], prevLayer, this.learningRate);
			prevLayer = this.hiddenLayers[i];
		}

		this.outputLayer = new SoftMaxLayer(this.outputSize, prevLayer, this.learningRate);
	}

	// initialize
	this.reset();

	function collapseLabeledVector (labeledVector, labelToIndex) {
		var collapsed = new Array(Object.keys(labelToIndex).length);
		Object.keys(labeledVector).forEach(function (label) {
			collapsed[labelToIndex[label]] = labeledVector[label];
		});
		return collapsed;
	}

	function labelArray (arr, labelToIndex) {
		var labeling = {};
		Object.keys(labelToIndex).forEach(function (label) {
			labeling[label] = arr[labelToIndex[label]];
		});
		return labeling;
	}

	this.train = function (labeledInputSignals, labeledEmbeddingVectors, labeledOutputSignals) {
		var embeddings = collapseLabeledVector(labeledEmbeddingVectors, this.embeddingLabels);
		var inputSignals = collapseLabeledVector(labeledInputSignals, this.inputLabels);
		var outputSignals = collapseLabeledVector(labeledOutputSignals, this.outputLabels);
		this.inputLayer.setSignals(inputSignals);
		this.inputLayer.setEmbeddings(embeddings);
		this.outputLayer.forward();
		this.outputLayer.target(outputSignals);

		return {
			resultEmbeddings: labelArray(this.inputLayer.embeddings, this.embeddingLabels)
			// error after
			// error before
		};
	}

	this.predict = function (labeledInputSignals, labeledEmbeddingVectors) {
		var embeddings = collapseLabeledVector(labeledEmbeddingVectors, this.embeddingLabels);
		var inputSignals = collapseLabeledVector(labeledInputSignals, this.inputLabels);

		this.inputLayer.setSignals(inputSignals);
		this.inputLayer.setEmbeddings(embeddings);
		this.outputLayer.forward();

		return {
			outputs: labelArray(this.outputLayer.outputs, this.outputLabels)
		};
	}
}

module.exports = NeuralNetwork;
