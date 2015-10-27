var Activation = (function () {

	var sigmoid = { };

	sigmoid.func = function (x) {
		return 1 / (1 + Math.exp(-x));
	}

	sigmoid.deriv = function (x) {
		return x * (1 - x);
	}

	return {
		sigmoid: sigmoid
	}

})();


module.exports = Activation;
