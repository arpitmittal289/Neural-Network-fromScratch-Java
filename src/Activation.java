package assignment3.NN.self;
import static java.lang.Math.exp;
import static java.lang.Math.log;
public class Activation {

    private final String name;
    private Function fn;
    private Function dFn;

    public Activation(String name) {
        this.name = name;
    }

    public Activation(String name, Function fn, Function dFn) {
        this.name = name;
        this.fn = fn;
        this.dFn = dFn;
    }

    public Matrix fn(Matrix in) {
        return in.map(fn);
    }

    public Matrix dFn(Matrix out) {
        return out.map(dFn);
    }

    public Matrix dCdI(Matrix out, Matrix dCdO) {
        return dCdO.multiplyElements(dFn(out));
    }

    public String getName() {
        return name;
    }


    public static Activation ReLU = new Activation(
        "ReLU",
        x -> x <= 0 ? 0 : x,                // fn
        x -> x <= 0 ? 0 : 1                 // dFn
    );

    public static Activation Leaky_ReLU = new Activation(
        "Leaky_ReLU",
        x -> x <= 0 ? 0.01 * x : x,         // fn
        x -> x <= 0 ? 0.01 : 1              // dFn
    );


    public static Activation Sigmoid = new Activation(
        "Sigmoid",
        Activation::sigmoidFn,                      // fn
        x -> sigmoidFn(x) * (1.0 - sigmoidFn(x))    // dFn
    );


    public static Activation Softplus = new Activation(
        "Softplus",
        x -> log(1.0 + exp(x)),             // fn
        Activation::sigmoidFn               // dFn
    );

    public static Activation Identity = new Activation(
        "Identity",
        x -> x,                             // fn
        x -> 1                              // dFn
    );


	/*
	 * public static Activation Softmax = new Activation("Softmax") {
	 * 
	 * @Override public Matrix fn(Matrix in) { double[] data = in.getData(); double
	 * sum = 0; double max = in.max(); // Trick: translate the input by largest
	 * element to avoid overflow. for (double a : data) sum += exp(a - max);
	 * 
	 * double finalSum = sum; return in.map(a -> exp(a - max) / finalSum); }
	 * 
	 * @Override public Matrix dCdI(Matrix out, Matrix dCdO) { double x =
	 * out.multiplyElements(dCdO).sumElements(); Matrix sub = dCdO.subtract(x);
	 * return out.multiplyElements(sub); } };
	 */


    // --------------------------------------------------------------------------

    private static double sigmoidFn(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

}

