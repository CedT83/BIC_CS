package neuralNetwork;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;


public class Perceptron implements Serializable {

    /**
	 * 
	 */
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

    //Copy constructor for this class
    public Perceptron copy() {
        Perceptron copy = new Perceptron(this.name);

        NeuralLayer previousNeuralLayer = null;
        for(NeuralLayer layer : layers) {

            NeuralLayer layerCopy;

            if(layer.HasBias()) {
                Neuron bias = layer.getNeurons().get(0);
                Neuron biasCopy = new Neuron(bias.getActivationStrategy().copy());
                biasCopy.setOutput(bias.getOutput());
                layerCopy = new NeuralLayer(null, biasCopy);
            }

            else {
                layerCopy = new NeuralLayer();
            }

            layerCopy.setPreviousNeuralLayer(previousNeuralLayer);

            int biasCount = layerCopy.HasBias() ? 1 : 0;

            for(int i = biasCount; i < layer.getNeurons().size(); i++) {
                Neuron neuron = layer.getNeurons().get(i);

                Neuron neuronCopy = new Neuron(neuron.getActivationStrategy().copy());
                neuronCopy.setOutput(neuron.getOutput());
                neuronCopy.setError(neuron.getError());

                if(neuron.getInputs().size() == 0) {
                    layerCopy.AddNeuron(neuronCopy);
                }

                else {
                    double[] weights = neuron.getWeights();
                    layerCopy.AddNeuron(neuronCopy, weights);
                }
            }

            copy.addNeuralLayer(layerCopy);
            previousNeuralLayer = layerCopy;
        }

        return copy;
    }

    //We use this method to add a layer to the network
    public void addNeuralLayer(NeuralLayer layer) {
        layers.add(layer);

        if(layers.size() == 1) {
            input = layer;
        }

        if(layers.size() > 1) {
            //clear the output flag on the previous output layer, but only if we have more than 1 layer
            NeuralLayer previousNeuralLayer = layers.get(layers.size() - 2);
            previousNeuralLayer.setNextNeuralLayer(layer);
        }

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

    //Returns the perceptron's name
    public String getName() {
        return name;
    }

    //We use this method to get the output of the network 
    public double[] getOutput() {

        double[] outputs = new double[output.getNeurons().size()];

        for(int i = 1; i < layers.size(); i++) {
            NeuralLayer layer = layers.get(i);
            layer.FeedForward();
        }

        int i = 0;
        for(Neuron neuron : output.getNeurons()) {
            outputs[i] = neuron.getOutput();
            i++;
        }

        return outputs;
    }

    //Basically, returns the list of layers in the network
    public List<NeuralLayer> getNeuralLayers() {
        return layers;
    }

    //This method resets all the weights of the network to a random value
    public void reset() {
        for(NeuralLayer layer : layers) {
            for(Neuron neuron : layer.getNeurons()) {
                for(Synapse synapse : neuron.getInputs()) {
                    synapse.setWeight((Math.random() * 1) - 0.5);
                }
            }
        }
    }

    //We retrieve the weights of each Neuron of each layer for the network
    public double[] getWeights() {

        List<Double> weights = new ArrayList<Double>();

        for(NeuralLayer layer : layers) {

            for(Neuron neuron : layer.getNeurons()) {

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

    //We use another Perceptron to define the weights of the current one
    public void copyWeightsFrom(Perceptron sourcePerceptron) {
        if(layers.size() != sourcePerceptron.layers.size()) {
            throw new IllegalArgumentException("Cannot copy weights. Number of layers do not match (" + sourcePerceptron.layers.size() + " in source versus " + layers.size() + " in destination)");
        }

        int i = 0;
        for(NeuralLayer sourceNeuralLayer : sourcePerceptron.layers) {
            NeuralLayer destinationNeuralLayer = layers.get(i);

            if(destinationNeuralLayer.getNeurons().size() != sourceNeuralLayer.getNeurons().size()) {
                throw new IllegalArgumentException("Number of neurons do not match in layer " + (i + 1) + "(" + sourceNeuralLayer.getNeurons().size() + " in source versus " + destinationNeuralLayer.getNeurons().size() + " in destination)");
            }

            int j = 0;
            for(Neuron sourceNeuron : sourceNeuralLayer.getNeurons()) {
                Neuron destinationNeuron = destinationNeuralLayer.getNeurons().get(j);

                if(destinationNeuron.getInputs().size() != sourceNeuron.getInputs().size()) {
                    throw new IllegalArgumentException("Number of inputs to neuron " + (j + 1) + " in layer " + (i + 1) + " do not match (" + sourceNeuron.getInputs().size() + " in source versus " + destinationNeuron.getInputs().size() + " in destination)");
                }

                int k = 0;
                for(Synapse sourceSynapse : sourceNeuron.getInputs()) {
                    Synapse destinationSynapse = destinationNeuron.getInputs().get(k);

                    destinationSynapse.setWeight(sourceSynapse.getWeight());
                    k++;
                }

                j++;
            }

            i++;
        }
    }

    //We print in a file the states of the network
    public void persist() {
        String fileName = name.replaceAll(" ", "") + "-" + new Date().getTime() +  ".net";
        System.out.println("Writing trained neural network to file " + fileName);

        ObjectOutputStream objectOutputStream = null;

        try {
            objectOutputStream = new ObjectOutputStream(new FileOutputStream(fileName));
            objectOutputStream.writeObject(this);
        }

        catch(IOException e) {
            System.out.println("Could not write to file: " + fileName);
            e.printStackTrace();
        }

        finally {
            try {
                if(objectOutputStream != null) {
                    objectOutputStream.flush();
                    objectOutputStream.close();
                }
            }

            catch(IOException e) {
                System.out.println("Could not write to file: " + fileName);
                e.printStackTrace();
            }
        }
    }
}
