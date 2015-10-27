var Brain = {
	NeuralNetwork: require('./neuralnetwork'),
	Utils: {
		Vector: require('./utils/vector'),
		Activation: require('./utils/activation')
	},
	Layers: {
		EmbeddingLayer: require('./layers/embedding'),
		HiddenLayer: require('./layers/hidden'),
		SoftMaxLayer: require('./layers/softmax')
	}
}

module.exports = Brain;
