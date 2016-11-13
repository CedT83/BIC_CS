package neuralNetwork;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Perceptron implements Serializable {

	//Without this line, we get a warning at compilation
	private static final long serialVersionUID = 1L;
	//We give a name for the perceptron
	private String name;
	//We define the list of layers composing the perceptron
    private List<NeuralLayer> layers;
    //We define the input layer
    private NeuralLayer input;
    //We define the output layer
    private NeuralLayer output;

    //Using this constructor we build a perceptron by giving it a name
    public Perceptron(String name) {
        this.name = name;
        layers = new ArrayList<NeuralLayer>();
    }


    //We use this method to add a layer to the network
    public void addNeuralLayer(NeuralLayer layer) {
        layers.add(layer);

        //If there is no layer yet we add the given in parameter as the input layer
        if(layers.size() == 1) {
            input = layer;
        }

        if(layers.size() > 1) {
            //clear the output flag on the previous output layer, but only if we have more than 1 layer
            NeuralLayer previousNeuralLayer = layers.get(layers.size() - 2);
            previousNeuralLayer.setNextNeuralLayer(layer);
        }

        //We define the ouput layer as the last layer added
        output = layers.get(layers.size() - 1);
    }

    //We use this method to set the layer's inputs using an array of double
    public void setInputs(double[] inputs) {
        if(input != null) {

            int biasCount = input.HasBias() ? 1 : 0;
            //If the number of inputs does not match with the number of inputs, we throw an Exception
            if(input.getNeurons().size() - biasCount != inputs.length) {
                throw new IllegalArgumentException("The number of inputs must equal the number of neurons in the input layer");
            }

            else {
                List<Neuron> neurons = input.getNeurons();
                for(int i = biasCount; i < neurons.size(); i++) {
                    neurons.get(i).setOutput(inputs[i - biasCount]);
                }
            }
        }
    }

    //We use this method to get the output of the network 
    public double[] getOutput() {

        double[] outputs = new double[output.getNeurons().size()];

        //For all the layers, we calculate their output 
        for(int i = 1; i < layers.size(); i++) {
            NeuralLayer layer = layers.get(i);
            //Propagate the input of current layer to input of next layer
            layer.FeedForward();
        }

        int i = 0;
        //With all the ouputs of the neurons from the output layer we create an array
        for(Neuron neuron : output.getNeurons()) {
            outputs[i] = neuron.getOutput();
            i++;
        }
        //We return the array containing the ouput of the last layer
        return outputs;
    }

    //Basically, returns the list of layers in the network
    public List<NeuralLayer> getNeuralLayers() {
        return layers;
    }

    //This method resets all the weights of the network to a random value
    public void reset() {
    	//for all layers
        for(NeuralLayer layer : layers) {
        	//for all neurons in a layer
            for(Neuron neuron : layer.getNeurons()) {
            	//for all weights on a neuron
                for(Synapse synapse : neuron.getInputs()) {
                	//set the value between -0.5 and 0.5
                    synapse.setWeight((Math.random() * 1) - 0.5);
                }
            }
        }
    }

    //We retrieve the weights of each Neuron of each layer for the network
    public double[] getWeights() {

        List<Double> weights = new ArrayList<Double>();
      //for all layers
        for(NeuralLayer layer : layers) {
        	//for all neurons in a layer
            for(Neuron neuron : layer.getNeurons()) {
            	//for all weights on a neuron
                for(Synapse synapse: neuron.getInputs()) {
                    weights.add(synapse.getWeight());
                }
            }
        }

        double[] allWeights = new double[weights.size()];

        int i = 0;
        for(Double weight : weights) {
            allWeights[i] = weight;
            i++;
        }

        return allWeights;
    }

}
