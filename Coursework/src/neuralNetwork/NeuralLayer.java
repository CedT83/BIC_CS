package neuralNetwork;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;


public class NeuralLayer implements Serializable {

    /**
	 * 
	 */
	//Without this line, we get a warning at compilation
	private static final long serialVersionUID = 1L;
	//The list of neurons which compose the layer
	private List<Neuron> neurons;
	//The reference to the object that models the previous layer
    private NeuralLayer previousNeuralLayer;
    //The reference to the object that models the next layer
    private NeuralLayer nextNeuralLayer;
    //Prevent from getting a weighted sum equals to 0
    private Neuron bias;

    //Builds a layer without any parameter, put the previous layer's reference to null
    public NeuralLayer() {
        neurons = new ArrayList<Neuron>();
        previousNeuralLayer = null;
    }

    //Builds a layer by precising the previous layer
    public NeuralLayer(NeuralLayer previousNeuralLayer) {
    	//We call the default constructor defined above
        this();
        this.previousNeuralLayer = previousNeuralLayer;
    }

    //Another constructor with the previous layer and the bias Neuron
    public NeuralLayer(NeuralLayer previousNeuralLayer, Neuron bias) {
        this(previousNeuralLayer);
        this.bias = bias;
        neurons.add(bias);
    }

    //Return the lst of the Neurons composing the layer
    public List<Neuron> getNeurons() {
        return this.neurons;
    }

    //This method adds a neuron to the list
    public void AddNeuron(Neuron neuron) {

        neurons.add(neuron);
        //We ensure that there is a previous layer to add the neuron
        if(previousNeuralLayer != null) {
            for(Neuron previousNeuralLayerNeuron : previousNeuralLayer.getNeurons()) {
                neuron.AddInput(new Synapse(previousNeuralLayerNeuron, (Math.random() * 1) - 0.5)); //initialize with a random weight between -1 and 1
            }
        }
    }

    //This method adds a neuron to the list and precise the weights between the new neuron and the linked neurons
    public void AddNeuron(Neuron neuron, double[] weights) {

        neurons.add(neuron);

        if(previousNeuralLayer != null) {

            if(previousNeuralLayer.getNeurons().size() != weights.length) {
                throw new IllegalArgumentException("The number of weights supplied must be equal to the number of neurons in the previous layer");
            }

            else {
                List<Neuron> previousNeuralLayerNeurons = previousNeuralLayer.getNeurons();
                for(int i = 0; i < previousNeuralLayerNeurons.size(); i++) {
                    neuron.AddInput(new Synapse(previousNeuralLayerNeurons.get(i), weights[i]));
                }
            }

        }
    }

    //For each neuron in the layer we call the activate function to go forward in the network
    public void FeedForward() {

        int biasCount = HasBias() ? 1 : 0;

        for(int i = biasCount; i < neurons.size(); i++) {
            neurons.get(i).Activate();
        }
    }

    //Returns the reference to the previous layer
    public NeuralLayer getPreviousNeuralLayer() {
        return previousNeuralLayer;
    }

    //Sets the previous layer to the value passed in parameter
    void setPreviousNeuralLayer(NeuralLayer previousNeuralLayer) {
        this.previousNeuralLayer = previousNeuralLayer;
    }

    //Returns the reference to the next layer
    public NeuralLayer getNextNeuralLayer() {
        return nextNeuralLayer;
    }

    //Sets the next layer to the value passed in parameter
    void setNextNeuralLayer(NeuralLayer nextNeuralLayer) {
        this.nextNeuralLayer = nextNeuralLayer;
    }

    //Tests if we are in the final layer
    public boolean IsOutputNeuralLayer() {
        return nextNeuralLayer == null;
    }

    //Tests if a Bias neuron exists in the layer
    public boolean HasBias() {
        return bias != null;
    }
}