package assignment3.self.neural.network;

public class TanhActivationLayer implements Layer {
	private Matrix input = null;
	private Matrix output = null;

	public Matrix activation(Matrix input) {
		Matrix temp = new Matrix(input.rows, input.cols);
		for (int i = 0; i < input.rows; i++) {
			for (int j = 0; j < input.cols; j++)
				temp.data[i][j] = Math.tanh(input.data[i][j]);
		}

		return temp;
	}

	public Matrix activationPrime(Matrix input) {
		Matrix temp = new Matrix(input.rows, input.cols);
		for (int i = 0; i < input.rows; i++) {
			for (int j = 0; j < input.cols; j++)
				temp.data[i][j] = 1 - Math.pow(Math.tanh(input.data[i][j]), 2);
		}

		return temp;
	}

	@Override
	public Matrix forwardPropagation(Matrix input) {
		this.input = input;
		output = this.activation(input);
		return output;
	}

	public Matrix backwardPropagation(Matrix outputerror, double lr) {
		return activationPrime(input).multiplyElements(outputerror);
	}

}
