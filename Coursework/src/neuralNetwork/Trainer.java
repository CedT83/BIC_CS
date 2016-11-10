package neuralNetwork;

//This class is used to train the neural network
public class Trainer {

	//We define an array of inputs for the neural network
    private double[][] inputs;
    //We define an array of expected outputs for the neural network
    private double[][] outputs;

    //Constructor using the inputs & outputs
    public Trainer(double[][] inputs, double[][] outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    //Returns the inputs stored in the attribute
    public double[][] getInputs() {
        return inputs;
    }

    //Returns the outputs stored in the attribute
    public double[][] getOutputs() {
        return outputs;
    }
}
