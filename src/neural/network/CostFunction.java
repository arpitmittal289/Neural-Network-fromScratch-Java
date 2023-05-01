package assignment3.self.neural.network;

public class CostFunction {

	public static Matrix loss(double targetVal, double outputVal, String lossType) {
		Matrix error = new Matrix(1, 1);
		if (lossType.equals("mse")) {
			error.data[0][0] = Math.pow(outputVal - targetVal, 2);
		}
		if (lossType.equals("crossentropy")) {
			if (targetVal == 1) {
				error.data[0][0] = (-1) * (Math.log(outputVal) / Math.log(2));
			} else {
				error.data[0][0] = (-1) * (Math.log(1 - outputVal) / Math.log(2));
			}
		}
		return error;
	}

	public static Matrix derivativeloss(double targetVal, double outputVal, String lossType) {
		Matrix error = new Matrix(1, 1);
		if (lossType.equals("mse")) {
			error.data[0][0] = 2 * (outputVal - targetVal);
		}
		if (lossType.equals("crossentropy")) {
			error.data[0][0] = ((-1) * (targetVal / outputVal) + ((1) * (1 - targetVal) / (1 - outputVal)));
		}
		return error;
	}
}
