var Brain = require('.');
var Utils = Brain.Utils;

var nn = new Brain.NeuralNetwork({
	inputVectorSize: 0,
	embeddingSizes: [5],
	hiddenLayerSizes: [3],
	outputSize: 2,
	embeddingLabels: ["A:B"],
	outputLabels: ["on", "off"],
	learningRate: 0.8
});

var em_0_0 = Utils.Vector.random(5);
var em_0_1 = Utils.Vector.random(5);
var em_1_0 = Utils.Vector.random(5);
var em_1_1 = Utils.Vector.random(5);

for (var i = 0; i < 100; i++) {
	em_0_0 = nn.train({}, {"A:B": em_0_0}, {"on": 0.0, "off": 1.0}).resultEmbeddings["A:B"];
	em_0_1 = nn.train({}, {"A:B": em_0_1}, {"on": 1.0, "off": 0.0}).resultEmbeddings["A:B"];
	em_1_0 = nn.train({}, {"A:B": em_1_0}, {"on": 1.0, "off": 0.0}).resultEmbeddings["A:B"];
	em_1_1 = nn.train({}, {"A:B": em_1_1}, {"on": 0.0, "off": 1.0}).resultEmbeddings["A:B"];
}

console.log(nn.predict({}, {"A:B": em_0_0}));
console.log(nn.predict({}, {"A:B": em_0_1}));
console.log(nn.predict({}, {"A:B": em_1_0}));
console.log(nn.predict({}, {"A:B": em_1_1}));
console.log(em_0_0);
console.log(em_0_1);
console.log(em_1_0);
console.log(em_1_1);

