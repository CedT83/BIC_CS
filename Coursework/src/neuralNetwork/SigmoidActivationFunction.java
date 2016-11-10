package neuralNetwork;

import java.io.Serializable;

//Must implement the ActivationFunction interface to ensure that we respect the requirements of an activation function defined by the interface
public class SigmoidActivationFunction implements ActivationFunction, Serializable {

	//Without this line, we get a warning at compilation
	private static final long serialVersionUID = 1L;

	//Calculate the result of the activation function using the weighed sum
	public double activate(double weightedSum) {
		//Formula for a sigmoid
        return 1.0 / (1 + Math.exp(-1.0 * weightedSum));
    }

	//Calculate the derivative for the sigmoid function
    public double derivative(double weightedSum) {
        return weightedSum * (1.0 - weightedSum);
    }

    //Copy constructor for this class
    public SigmoidActivationFunction copy() {
        return new SigmoidActivationFunction();
    }
}
