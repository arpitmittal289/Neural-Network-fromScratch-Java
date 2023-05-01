package assignment3.self.neural.network;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Network {
	static List<Layer> layers = new ArrayList<>();
	static String networkLossType = "mse";

	public void add(Layer layer) {
		layers.add(layer);
	}

	public void lossFun(String lossType) {
		networkLossType = lossType;
	}

	public List<Double> predict(List<double[]> inputData) {
		int inputSize = inputData.size();
		List<Double> result = new ArrayList<Double>();

		for (int i = 0; i < inputSize; i++) {
			Matrix output = Matrix.fromArrayAxisX(inputData.get(i));
			for (Layer layer : layers) {
				output = layer.forwardPropagation(output);
			}
			result.add(output.data[0][0]);
		}
		return result;
	}

	public void fit(List<double[]> trainDatapoints, List<Double> trainLabelPoints, int epochs, double lr,
			int batchSize) {
		int inputSize = trainDatapoints.size();

		List<Layer> revLayers = new ArrayList<Layer>(layers);
		Collections.reverse(revLayers);

		for (int e = 0; e < epochs; e++) {
			if (e % 5 == 0) {
				System.out.println("Epoch : " + e);
			}
			for (int i = 0; i < inputSize; i = i + batchSize) {
				for (int b = 0; b < batchSize; b++) {
					Matrix output = Matrix.fromArrayAxisX(trainDatapoints.get(i));
					for (Layer layer : layers) {
						output = layer.forwardPropagation(output);
					}
					Matrix cost = CostFunction.derivativeloss(trainLabelPoints.get(i), output.data[0][0],
							networkLossType);
					for (Layer layer : revLayers) {
						cost = layer.backwardPropagation(cost, lr);
					}
				}
				for (Layer layer : layers) {
					layer.updateWeightsAndBias();
				}
			}
		}

	}
}
