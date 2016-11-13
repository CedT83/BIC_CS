 package neuralNetwork;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;


public class Neuron implements Serializable 
{

	//Without this line, we get a warning at compilation
	private static final long serialVersionUID = 1L;
	
	//Each neuron has one or several link(s) to other neuron(s)from the previous layer
	private List<Synapse> inputs;
	
	//For each neuron we define its activation function
    private ActivationFunction activationFunction;
    
    //The output of the neuron is stored here
    private double output;
    
    //We defined the activation function, we also define its derivative
    private double derivative;
    
    //This variable contains the weighted sum of all the inputs
    private double weightedSum;
    
    //This variable is used for the backpropagation
    private double error;

    //Constructor of the class with argument : activation function
    public Neuron(ActivationFunction activationFunction)
    {
        inputs = new ArrayList<Synapse>();
        this.activationFunction = activationFunction;
        error = 0;
    }

    //We add another synapse (neuron plus the weight for the link) to our list of inputs
    public void AddInput(Synapse input) 
    {
        inputs.add(input);
    }

    //Basically, we retrieve the list of all the synapses for this neuron
    public List<Synapse> getInputs() 
    {
        return this.inputs;
    }

    //We retrieve an array containing all the weights of this neuron
    public double[] getWeights() 
    {
        double[] weights = new double[inputs.size()];

        int i = 0;
        for(Synapse synapse : inputs) {
            weights[i] = synapse.getWeight();
            i++;
        }
        return weights;
    }

    //We calculate the weighted sum using the inputs (synapses) and the weights of those synapses
    private void CalculateWeightedSum() 
    {
        weightedSum = 0;
        //We do it for all the synapses the neuron is linked to
        for(Synapse synapse : inputs) {
            weightedSum += synapse.getWeight() * synapse.getSourceNeuron().getOutput();
        }
    }

    //We determine if the neuron is firing or not using the activation function
    public void Activate() 
    {
    	//Calculate the weighted sum
        CalculateWeightedSum();
        //We calculate the output using the activation function
        output = activationFunction.activate(weightedSum);
        derivative = activationFunction.derivative(output);
    }

    //We just get the output of the neuron (result given by the activation function)
    public double getOutput() 
    {
        return this.output;
    }

    // This is the mutator to set the output variable to a certain value
    public void setOutput(double output) 
    {
        this.output = output;
    }

    //This is the accessor to the variable derivative
    public double getDerivative() 
    {
        return this.derivative;
    }

    //This function returns the activation function defined for this neuron
    public ActivationFunction getActivationStrategy() 
    {
        return activationFunction;
    }

    //We get the value of the variable error
    public double getError() 
    {
        return error;
    }

    //We set the value of the variable error to the value passed in parameter
    public void setError(double error) 
    {
        this.error = error;
    }
}
