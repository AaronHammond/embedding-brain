var Vector = (function () {

	var that = {};

	that.random = function (size) {
		var v =  new Array(size);
		for (var i = 0; i < size; i++) {
			v[i] = Math.random();
		}
		return v;
	}

	that.zeros = function (size) {
		var v = new Array(size);
		for (var i = 0; i < size; i++) {
			v[i] = 0;
		}
		return v;
	}

	that.normalize = function (v) {
		var norm = v.reduce(function (prev, curr) { return prev + curr }, 0);
		return v.map(function (val) { return val / norm; });
	}

	return that; 

})();

module.exports = Vector;
