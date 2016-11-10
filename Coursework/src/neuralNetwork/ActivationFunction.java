package neuralNetwork;

//This interface precises the methods an activation function must implement
public interface ActivationFunction {
    double activate(double weightedSum);
    double derivative(double weightedSum);
    ActivationFunction copy();
}
