function sigmoid(x) {
	return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
	return y * (1 - y);
}

class NeuralNetwork {
	
	constructor(input_nodes, hidden_nodes, output_nodes){
		this.input_nodes = input_nodes;
		this.hidden_nodes = hidden_nodes;
		this.output_nodes = output_nodes;

		this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
		this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);

		this.weights_ih.randomize();
		this.weights_ho.randomize();

		this.bias_h = new Matrix(this.hidden_nodes, 1);
		this.bias_o = new Matrix(this.output_nodes, 1);

		this.bias_h.randomize();
		this.bias_o.randomize();

		this.learning_rate = 0.1;
	}

	feedforward(input_array) {
		// Convert input array to matrix
		let inputs = Matrix.fromArray(input_array);

        // Generating the Hidden Outputs
		let hidden = Matrix.multiply(this.weights_ih, inputs);
		hidden.add(this.bias_h);
		// Activation function
		hidden.map(sigmoid);

        // Generating the Output's Output
		let output = Matrix.multiply(this.weights_ho, hidden);
		output.add(this.bias_o);
		// Activation function
		output.map(sigmoid);

		// Return the output as an array
		return output.toArray();
	}

	train(input_array, target_array){
		// Convert input array to matrix
		let inputs = Matrix.fromArray(input_array);

		 // Generating the Hidden Outputs
		let hidden = Matrix.multiply(this.weights_ih, inputs);
		hidden.add(this.bias_h);
		// Activation function
		hidden.map(sigmoid);

        // Generating the Output's Output
		let outputs = Matrix.multiply(this.weights_ho, hidden);
		outputs.add(this.bias_o);
		// Activation function
		outputs.map(sigmoid);

		// Convert array to matrix object
		let targets = Matrix.fromArray(target_array);

		// Calculate the error
		let output_errors = Matrix.subtract(targets, outputs);

		// Calculate output gradient 
		let output_gradient = Matrix.map(outputs, dsigmoid);
		output_gradient.multiply(output_errors);
		output_gradient.multiply(this.learning_rate);

		// Calculate hidden->output deltas
		let hidden_T = Matrix.transpose(hidden);
		let weights_ho_deltas = Matrix.multiply(output_gradient, hidden_T);

        // Adjust the weights by deltas
		this.weights_ho.add(weights_ho_deltas);
		// Adjust the bias by deltas(gradient)
		this.bias_o.add(output_gradient);

        // Calculate the hidden layer errors
		let weights_ho_t = Matrix.transpose(this.weights_ho);
		let hidden_errors = Matrix.multiply(weights_ho_t, output_errors);

        // Calculate hidden gradient
		let hidden_gradient = Matrix.map(hidden, dsigmoid);
		hidden_gradient.multiply(hidden_errors);
		hidden_gradient.multiply(this.learning_rate);

		// Calculate input->hidden deltas
		let inputs_T = Matrix.transpose(inputs);
		let weights_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

        // Adjust the weights by deltas
		this.weights_ih.add(weights_ih_deltas);
		// Adjust the bias by deltas(gradient)
		this.bias_h.add(hidden_gradient);
	}
}