package neuralNetwork;

import java.io.Serializable;


public class Synapse implements Serializable {

	//Without this line, we get a warning at compilation
	private static final long serialVersionUID = 1L;
	//This attribute corresponds to the Neuron linked to the synapse
	private Neuron sourceNeuron;
	//This is the weight attributed to the link between sourceNeuron and the synapse
    private double weight;

    //Builds the synapse with the Neuron linked and its weight attributed as parameters
    public Synapse(Neuron sourceNeuron, double weight) 
    {
        this.sourceNeuron = sourceNeuron;
        this.weight = weight;
    }

    //Returns the Neuron linked to the synapse
    public Neuron getSourceNeuron() 
    {
        return sourceNeuron;
    }

    //Returns the weight of the synapse
    public double getWeight() 
    {
        return weight;
    }
    
    //Sets the weight of the synapse
    public void setWeight(double weight) 
    {
        this.weight = weight;
    }
}
