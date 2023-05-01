package assignment3.NN.self;

import java.util.ArrayList;
import java.util.List;

public class CustomNeuralNetwork {
	Matrix weights_ho, bias_o;
	List<Matrix> weights_hh = new ArrayList<Matrix>();
	List<Matrix> bias_hh = new ArrayList<Matrix>();
	List<Matrix> hidden_output = new ArrayList<Matrix>();
	int hiddenSize = 0;
	double l_rate = 0.1;

	public CustomNeuralNetwork(int i, int[] h, int o) {
		weights_ho = new Matrix(o, h[h.length - 1]);
		for (int layerNum = 0; layerNum < h.length; layerNum++) {
			if (layerNum == 0) {
				weights_hh.add(new Matrix(h[layerNum], i));
			} else {
				weights_hh.add(new Matrix(h[layerNum], h[layerNum - 1]));
			}

			bias_hh.add(new Matrix(h[layerNum], 1));
		}

		double[] inputTemp = new double[i];
		hiddenSize = h.length;
		Matrix inputDemo = Matrix.fromArray(inputTemp);
		for (int layerNum = 0; layerNum < h.length; layerNum++) {
			if (layerNum == 0) {
				hidden_output.add(Matrix.multiply(weights_hh.get(layerNum), Matrix.fromArray(inputTemp))
						.add(bias_hh.get(layerNum)));
			} else {
				hidden_output.add(Matrix.multiply(weights_hh.get(layerNum), hidden_output.get(layerNum - 1))
						.add(bias_hh.get(layerNum)));
			}
		}

		bias_o = new Matrix(o, 1);

	}

	public Matrix forwardProp(double[] inputArr) {
		for (int layerNum = 0; layerNum < weights_hh.size(); layerNum++) {
			if (layerNum == 0) {
				hidden_output.add(Matrix.multiply(weights_hh.get(layerNum), Matrix.fromArray(inputArr))
						.add(bias_hh.get(layerNum)).sigmoid());
			} else {
				hidden_output.add(Matrix.multiply(weights_hh.get(layerNum), hidden_output.get(layerNum - 1))
						.add(bias_hh.get(layerNum)).sigmoid());
			}
		}

		Matrix output = Matrix.multiply(weights_ho, hidden_output.get(hiddenSize - 1));
		output.add(bias_o);
		output.sigmoid();

		return output;
	}

	public List<Double> predict(double[] inputArr) {
		return forwardProp(inputArr).toArray();
	}

	public void backwardProp(Matrix output, double[] targetArr, double[] inputArr) {
		Matrix input = Matrix.fromArray(inputArr);

		Matrix target = Matrix.fromArray(targetArr);
		Matrix error = Matrix.binaryDerivativeCrossEntropy(target, output);
		Matrix act_Error = Matrix.multiply(error, weights_ho);
		
		Matrix gradient = output.dsigmoid();
		gradient.multiply(error);
	
		Matrix hidden_T = Matrix.transpose(hidden_output.get(hiddenSize - 1));
		Matrix who_delta = Matrix.multiply(gradient, hidden_T);
		who_delta.multiply(l_rate);
		weights_ho.add(who_delta);
		gradient.multiply(l_rate);
		bias_o.add(gradient);

		
		for (int layerNum = hiddenSize - 1; layerNum >= 0; layerNum--) {
			Matrix who_T = null;
			if (layerNum == hiddenSize - 1) {
				who_T = Matrix.transpose(weights_ho);
			} else {
				who_T = Matrix.transpose(weights_hh.get(layerNum + 1));
			}

			Matrix h_gradient = hidden_output.get(layerNum).dsigmoid();
			h_gradient.multiply(act_Error);
			act_Error = Matrix.multiply(act_Error, weights_hh.get(layerNum));
			Matrix i_T = null;
			if (layerNum == 0) {
				i_T = Matrix.transpose(input);
			} else {
				i_T = Matrix.transpose(hidden_output.get(layerNum - 1));
			}
			Matrix wih_delta = Matrix.multiply(h_gradient, i_T);
			wih_delta.multiply(l_rate);
			weights_hh.get(layerNum).add(wih_delta);
			h_gradient.multiply(l_rate);
			bias_hh.get(layerNum).add(h_gradient);
		}
	}

	public void trainOnSingleData(double[] inputArr, double[] targetArr) {
		Matrix output = this.forwardProp(inputArr);
		backwardProp(output, targetArr, inputArr);
	}

	public void train(double[][] input, double[][] target, int epochs, double lr) {
		this.l_rate = lr;
		for (int i = 0; i < epochs; i++) {
			for (int sampleN = 0; sampleN < input.length; sampleN++) {
				this.trainOnSingleData(input[sampleN], target[sampleN]);
			}
		}
	}

	public void trainDataPoints(List<DataPoint> data, int epochs, double lr) {
		this.l_rate = lr;
		for (int i = 0; i < epochs; i++) {
			for (DataPoint point : data) {
				double[] input = { point.getX(), point.getY() };
				double[] output = { point.getLabel() };
				this.trainOnSingleData(input, output);
			}
		}
	}

	public List<Double> predictDataPoint(DataPoint point) {
		double[] input = { point.getX(), point.getY() };
		return this.predict(input);
	}

}
