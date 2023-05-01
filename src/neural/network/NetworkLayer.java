package assignment3.self.neural.network;

import java.util.ArrayList;
import java.util.List;

public class NetworkLayer implements Layer {
	public Matrix input = null;
	public Matrix output = null;
	public Matrix weights = null;
	public List<Matrix> batchDeltaWeights = new ArrayList<Matrix>();
	public List<Matrix> batchDeltaBias = new ArrayList<Matrix>();

	public Matrix getWeights() {
		return weights;
	}

	public void setWeights(Matrix weights) {
		this.weights = weights;
	}

	public Matrix bias = null;

	public NetworkLayer(int inputSize, int outputSize) {
		weights = new Matrix(inputSize, outputSize);
		bias = new Matrix(1, outputSize);
	}

	@Override
	public Matrix forwardPropagation(Matrix input) {
		this.input = input;
		output = Matrix.multiply(input, weights);
		output.add(bias);
		return output;
	}

	@Override
	public Matrix backwardPropagation(Matrix errorOutput, double lr) {

		Matrix errorInput = Matrix.multiply(errorOutput, Matrix.transpose(weights));
		Matrix errorWeights = Matrix.multiply(Matrix.transpose(input), errorOutput);

		errorWeights.multiply(lr);
		batchDeltaWeights.add(errorWeights);

		errorOutput.multiply(lr);
		batchDeltaBias.add(errorOutput);

		return errorInput;
	}

	@Override
	public void updateWeightsAndBias() {
		weights.subtract(Matrix.average(batchDeltaWeights));
		bias.subtract(Matrix.average(batchDeltaBias));
		batchDeltaWeights = new ArrayList<Matrix>();
		batchDeltaBias = new ArrayList<Matrix>();
	}

}
