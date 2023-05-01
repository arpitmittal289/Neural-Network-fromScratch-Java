package assignment3.self.neural.network;

public interface Layer {
	public Matrix forwardPropagation(Matrix input);

	public Matrix backwardPropagation(Matrix cost, double lr);
	
	default void updateWeightsAndBias() {
		
	};
}
