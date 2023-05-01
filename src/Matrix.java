package assignment3.NN.self;

import static java.lang.Math.exp;
import static java.lang.String.format;
import static java.lang.System.arraycopy;
import static java.util.Arrays.stream;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringJoiner;

public class Matrix {

	public final double[][] data;
	public final int rows;
	public final int cols;

	public Matrix(double[][] data) {
		this.data = data;
		rows = data.length;
		cols = data[0].length;
	}

	public Matrix(int rows, int cols) {
		this(new double[rows][cols]);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				data[i][j] = Math.random();
			}
		}
	}

	private void assertCorrectDimension(Matrix other) {
		if (rows != other.rows || cols != other.cols)
			throw new IllegalArgumentException(format("Matrix of different dim: Input is %d x %d, Vec is %d x %d", rows,
					cols, other.rows, other.cols));
	}

	public Matrix add(Matrix other) {
		assertCorrectDimension(other);

		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				data[y][x] += other.data[y][x];
			}
		}

		return this;
	}

	public Matrix add(double scaler) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				this.data[i][j] += scaler;
			}
		}

		return this;
	}

	public Matrix subtract(Matrix other) {
		assertCorrectDimension(other);
		for (int y = 0; y < rows; y++)
			for (int x = 0; x < cols; x++)
				data[y][x] -= other.data[y][x];

		return this;
	}

	public Matrix subtract(double scaler) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				this.data[i][j] -= scaler;
			}
		}

		return this;
	}

	public static Matrix subtract(Matrix a, Matrix b) {
		Matrix temp = new Matrix(a.rows, a.cols);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.cols; j++) {
				temp.data[i][j] = a.data[i][j] - b.data[i][j];
			}
		}
		return temp;
	}

	public Matrix transpose() {
		Matrix temp = new Matrix(cols, rows);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				temp.data[j][i] = data[i][j];
			}
		}
		return temp;
	}

	public static Matrix transpose(Matrix a) {
		Matrix temp = new Matrix(a.cols, a.rows);
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.cols; j++) {
				temp.data[j][i] = a.data[i][j];
			}
		}
		return temp;
	}

	public static Matrix multiply(Matrix a, Matrix b) {
		Matrix temp = new Matrix(a.rows, b.cols);
		for (int i = 0; i < temp.rows; i++) {
			for (int j = 0; j < temp.cols; j++) {
				double sum = 0;
				for (int k = 0; k < a.cols; k++) {
					sum += a.data[i][k] * b.data[k][j];
				}
				temp.data[i][j] = sum;
			}
		}
		return temp;
	}

	public void multiply(Matrix b) {
		Matrix temp = new Matrix(rows, b.cols);
		for (int i = 0; i < temp.rows; i++) {
			for (int j = 0; j < temp.cols; j++) {
				double sum = 0;
				for (int k = 0; k < cols; k++) {
					sum += data[i][k] * b.data[k][j];
				}
				this.data[i][j] = sum;
			}
		}
	}

	public Matrix multiplyElements(Matrix a) {
		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < a.cols; j++) {
				this.data[i][j] *= a.data[i][j];
			}
		}

		return this;

	}

	public Matrix multiply(double a) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				this.data[i][j] *= a;
			}
		}

		return this;

	}

	public int rows() {
		return rows;
	}

	public int cols() {
		return cols;
	}

	public double[][] getData() {
		return data;
	}

	public Matrix map(Function fn) {
		for (int y = 0; y < rows; y++)
			for (int x = 0; x < cols; x++)
				data[y][x] = fn.apply(data[y][x]);

		return this;
	}

	public Matrix fillFrom(Matrix other) {
		assertCorrectDimension(other);

		for (int y = 0; y < rows; y++) {
			if (cols >= 0) {
				arraycopy(other.data[y], 0, data[y], 0, cols);
			}
		}

		return this;
	}

	public double average() {
		return stream(data).flatMapToDouble(Arrays::stream).average().getAsDouble();
	}

	public double variance() {
		double avg = average();
		return stream(data).flatMapToDouble(Arrays::stream).map(a -> (a - avg) * (a - avg)).average().getAsDouble();
	}

	public Matrix copy() {
		Matrix m = new Matrix(rows, cols);
		for (int y = 0; y < rows; y++)
			if (cols >= 0)
				arraycopy(data[y], 0, m.data[y], 0, cols);

		return m;
	}

	@Override
	public String toString() {
		return new StringJoiner(", ", Matrix.class.getSimpleName() + "[", "]").add("data=" + Arrays.deepToString(data))
				.toString();
	}

	public static Matrix fromArray(double[] x) {
		Matrix temp = new Matrix(x.length, 1);
		for (int i = 0; i < x.length; i++)
			temp.data[i][0] = x[i];
		return temp;

	}

	public static Matrix fromArrayAxisX(double[] x) {
		Matrix temp = new Matrix(1, x.length);
		for (int i = 0; i < x.length; i++)
			temp.data[0][i] = x[i];
		return temp;

	}
	
	public List<Double> toArray() {
		List<Double> temp = new ArrayList<Double>();

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				temp.add(data[i][j]);
			}
		}
		return temp;
	}

	public Matrix sigmoid() {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++)
				this.data[i][j] = sigmoidFn(this.data[i][j]);
		}

		return this;

	}

	public Matrix dsigmoid() {
		Matrix temp = new Matrix(rows, cols);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++)
				temp.data[i][j] = sigmoidFn(this.data[i][j]) * (1 - sigmoidFn(this.data[i][j]));
		}
		return temp;

	}

	private static double sigmoidFn(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public static Matrix binaryCrossEntropy(Matrix target, Matrix output) {
		double targetVal = target.data[0][0];
		double outputVal = output.data[0][0];

		Matrix error = new Matrix(1, 1);
		if (targetVal == 1) {
			error.data[0][0] = (-1) * (Math.log(outputVal) / Math.log(2));
		} else {
			error.data[0][0] = (-1) * (Math.log(1 - outputVal) / Math.log(2));
		}

		return error;
	}

	public static Matrix binaryMSE(Matrix target, Matrix output) {
		double targetVal = target.data[0][0];
		double outputVal = output.data[0][0];

		Matrix error = new Matrix(1, 1);
		error.data[0][0] = Math.pow(outputVal - targetVal, 2);

		return error;
	}

	public static Matrix binaryDerivativeMSE(Matrix target, Matrix output) {
		double targetVal = target.data[0][0];
		double outputVal = output.data[0][0];

		Matrix error = new Matrix(1, 1);
		error.data[0][0] = 2 * (outputVal - targetVal);

		return error;
	}

	public static Matrix binaryDerivativeCrossEntropy(Matrix target, Matrix output) {
		double targetVal = target.data[0][0];
		double outputVal = output.data[0][0];

		Matrix error = new Matrix(1, 1);
		error.data[0][0] = ((-1)*(targetVal/outputVal)+((1)*(1-targetVal)/(1-outputVal)));
		return error;
	}

}
