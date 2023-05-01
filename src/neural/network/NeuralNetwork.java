package assignment3.self.neural.network;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
	static List<List<Integer>> routes = new ArrayList<>();

	public static void main(String[] args) throws IOException {

	}

	public static int findcheapestflight(int src, int des, int stops) {
		if (stops == -1) {
			return -1;
		}
		if (src == des) {
			return 0;
		}
		int result = Integer.MAX_VALUE;

		int tempPrice = Integer.MAX_VALUE;
		for (List<Integer> route : routes) {
			if (route.get(0) == src) {
				tempPrice = findcheapestflight(route.get(1), des, stops - 1);
			}
			if (tempPrice == -1) {
				continue;
			}
			tempPrice += route.get(2);
			result = Math.min(result, tempPrice);
		}

		if (tempPrice == Integer.MAX_VALUE) {
			return -1;
		}
		return result;

	}
}
