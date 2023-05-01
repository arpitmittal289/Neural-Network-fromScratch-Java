package assignment3.NN.self;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class NeuralNetwork {
	public static void main(String[] args) throws IOException {

		List<DataPoint> trainData = getDataFromInput(new File(args[0]), new File(args[1]));
		List<DataPoint> testData = getDataFromInput(new File(args[2]), null);

		int[] hidden_nodes = {8,6};

		CustomNeuralNetwork nn = new CustomNeuralNetwork(2, hidden_nodes, 1);

		List<DataPoint> postrainData = new ArrayList<DataPoint>();
		List<DataPoint> negtrainData = new ArrayList<DataPoint>();

		for (DataPoint trainDataPoint : trainData) {
			int target = trainDataPoint.getLabel() > 0.5 ? 1 : 0;
			if (target == 1) {
				postrainData.add(trainDataPoint);
			} else {
				negtrainData.add(trainDataPoint);
			}
		}

		int minSize = Math.min(postrainData.size(), negtrainData.size());

		List<DataPoint> finaltrainData = new ArrayList<DataPoint>();

		Random rn = new Random();
		for (int i = 0; i < postrainData.size(); i++) {
			finaltrainData.add(postrainData.get(rn.nextInt(postrainData.size())));
		}

		for (int i = 0; i < negtrainData.size(); i++) {
			finaltrainData.add(negtrainData.get(rn.nextInt(negtrainData.size())));
		}
		
		System.out.println("Training...");
		nn.trainDataPoints(trainData, 1000, 0.01);

		int correct0Count = 0;
		int correct1Count = 0;
		int total0Count = 0;
		int total1Count = 0;
		System.out.println("Train Loss-");
		for (DataPoint testDataPoint : finaltrainData) {
			List<Double> predicted = nn.predictDataPoint(testDataPoint);
			int predict = predicted.get(0) > 0.5 ? 1 : 0;
			int target = testDataPoint.getLabel() > 0.5 ? 1 : 0;
			if (target == 0) {
				total0Count++;
			} else {
				total1Count++;
			}
			if (predict == target) {
				if (predict == 0) {
					correct0Count++;
				} else {
					correct1Count++;
				}

			}
		}
		int total = trainData.size();
		System.out.println("Total : " + total);
		System.out.println("Total 0 : " + total0Count);
		System.out.println("Correct 0 : " + correct0Count);
		System.out.println("Total 1 : " + total1Count);
		System.out.println("Correct 1 : " + correct1Count);

		System.out.println("Test Loss-");
		correct0Count = 0;
		correct1Count = 0;
		total0Count = 0;
		total1Count = 0;
		for (DataPoint testDataPoint : testData) {
			List<Double> predicted = nn.predictDataPoint(testDataPoint);
			System.out.println(predicted.get(0));
			int predict = predicted.get(0) > 0.5 ? 1 : 0;
			int target = testDataPoint.getLabel() > 0.5 ? 1 : 0;
			if (target == 0) {
				total0Count++;
			} else {
				total1Count++;
			}
			if (predict == target) {
				if (predict == 0) {
					correct0Count++;
				} else {
					correct1Count++;
				}

			}
		}
		total = testData.size();
		System.out.println("Total : " + total);
		System.out.println("Total 0 : " + total0Count);
		System.out.println("Correct 0 : " + correct0Count);
		System.out.println("Total 1 : " + total1Count);
		System.out.println("Correct 1 : " + correct1Count);

	}

	private static List<DataPoint> getDataFromInput(File dataFile, File labelFile) throws FileNotFoundException {
		Scanner dataSc = new Scanner(dataFile);
		int i = 1;
		List<DataPoint> data = new ArrayList<DataPoint>();
		if (null != labelFile) {
			Scanner labelSc = new Scanner(labelFile);
			while (dataSc.hasNextLine() && labelSc.hasNextLine()) {
				String cordinateTuple = dataSc.nextLine();
				String label = labelSc.nextLine();
				String[] cordinateArr = cordinateTuple.split(",");
				DataPoint dp = new DataPoint();
				dp.setX(Double.parseDouble(cordinateArr[0]));
				dp.setY(Double.parseDouble(cordinateArr[1]));
				dp.setLabel(Double.parseDouble(label));
				data.add(dp);
			}
		} else {
			while (dataSc.hasNextLine()) {
				String cordinateTuple = dataSc.nextLine();
				String[] cordinateArr = cordinateTuple.split(",");
				DataPoint dp = new DataPoint();
				dp.setX(Double.parseDouble(cordinateArr[0]));
				dp.setY(Double.parseDouble(cordinateArr[1]));
				data.add(dp);
			}
		}
		return data;
	}

}
