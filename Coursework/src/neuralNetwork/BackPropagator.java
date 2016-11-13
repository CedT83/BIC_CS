package neuralNetwork;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BackPropagator {

	//Our neural network to train
    private Perceptron neuralNetwork;
    //Its learning rate
    private double learningRate;
    //Its momentum
    private double momentum;
    //Its characteristic time. Not used for the moment
    private double characteristicTime;

    //Constructor with parameters
    public BackPropagator(Perceptron neuralNetwork, double learningRate, double momentum, double characteristicTime) 
    {
        this.neuralNetwork = neuralNetwork;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.characteristicTime = characteristicTime;
    }
    
    //Method that thains our neural network
    public void train(TrainerGenerator generator, double errorThreshold) 
    {

    	//Initialization of local variables
        double error = 0.0;
        double sum = 0.0;
        double average = 25;
        int epoch = 1;
        int samples = 25;
        double[] errors = new double[samples];

        //We call the back propagate method 
        do {
            Trainer trainingData = generator.getTrainer();
            error = backPropagate(trainingData.getInputs(), trainingData.getOutputs());

            sum -= errors[epoch % samples];
            errors[epoch % samples] = error;
            sum += errors[epoch % samples];

            //We calculate the average for the error.
            if(epoch > samples) {
                average = sum / samples;
            }

            //We print the results of training 
            System.out.println("Error for epoch " + epoch + ": " + error + ". Average: " + average);
            epoch++;
        //Until the average error is lower than the minimal error defined for the neural network
        } while(average > errorThreshold);
    }

    public double backPropagate(double[][] inputs, double[][] expectedOutputs) 
    {

        double error = 0;

        Map<Synapse, Double> synapseNeuronDeltaMap = new HashMap<Synapse, Double>();

        //We do for all the row of the array containing the training data
        for (int i = 0; i < inputs.length; i++) {

            double[] input = inputs[i];
            double[] expectedOutput = expectedOutputs[i];

            List<NeuralLayer> layers = neuralNetwork.getNeuralLayers();

            neuralNetwork.setInputs(input);
            double[] output = neuralNetwork.getOutput();
            
            //First step of the backpropagation algorithm. Backpropagate errors from the output layer all the way up
            //to the first hidden layer
            //For all the layers
            for (int j = layers.size() - 1; j > 0; j--) {
                NeuralLayer layer = layers.get(j);

                //For all the neurons in the layer
                for (int k = 0; k < layer.getNeurons().size(); k++) {
                    Neuron neuron = layer.getNeurons().get(k);
                    double neuronError = 0;

                    //Test for the ouput layer, to apply the right formula
                    if (layer.IsOutputNeuralLayer()) {
             // modified --------------------------------------------------------------------------------
                    	//calculate the rror using all the neurons in the layer
                    	for(int l = 0; l < layer.getNeurons().size(); l++) 
                    	{
                    			neuronError = neuron.getDerivative() * (output[k] - expectedOutput[k]);
                    	} 
                    	
            // fin modified ------------------------------------------------------------------------------
                    //If the layer is not an output layer we apply another formula
                    } else {
                        neuronError = neuron.getDerivative();

                        double sum = 0;
                        List<Neuron> downstreamNeurons = layer.getNextNeuralLayer().getNeurons();
                        for (Neuron downstreamNeuron : downstreamNeurons) {

                            int l = 0;
                            boolean found = false;
                            //We update the weights for all the neurons up-connected to the current one
                            while (l < downstreamNeuron.getInputs().size() && !found) {
                                Synapse synapse = downstreamNeuron.getInputs().get(l);

                                if (synapse.getSourceNeuron() == neuron) {
                                    sum += (synapse.getWeight() * downstreamNeuron.getError());
                                    found = true;
                                }

                                l++;
                            }
                        }
                        neuronError *= sum;
                    }

                    neuron.setError(neuronError);
                }
            }

            //Second step of the backpropagation algorithm. Using the errors calculated above, update the weights of the
            //network
            //For all layers in the NN
            for(int j = layers.size() - 1; j > 0; j--) {
                NeuralLayer layer = layers.get(j);
                //For all neurons in a layer
                for(Neuron neuron : layer.getNeurons()) {
                	//For all previous neurons of the current one
                    for(Synapse synapse : neuron.getInputs()) {
                    	//We calculate the delta for the error
                        double delta = learningRate * neuron.getError() * synapse.getSourceNeuron().getOutput(); 

                        //If there is a neuron before the current one
                        if(synapseNeuronDeltaMap.get(synapse) != null) {
                            double previousDelta = synapseNeuronDeltaMap.get(synapse);
                            //We use the current delta and the previous one for more accuracy
                            delta += momentum * previousDelta;
                        }
                        
                        //We modify the weight between the neurons
                        synapseNeuronDeltaMap.put(synapse, delta);
                        synapse.setWeight(synapse.getWeight() - delta);
                    }
                }
            }

            output = neuralNetwork.getOutput();
            error += error(output, expectedOutput);
        }

        return error;
    }

    //Method to calculate the mean squared error 
    public double error(double[] actual, double[] expected) 
    {
    	//We throw an error if expected and real ouput have not the same length
        if (actual.length != expected.length) {
            throw new IllegalArgumentException("The lengths of the actual and expected value arrays must be equal");
        }

        double sum = 0;

        //For MSE. 
        for (int i = 0; i < expected.length; i++) {
            sum += Math.pow(expected[i] - actual[i], 2);
        }
        //Error at the power of 2 divided by n (number of inputs of the network here it is 2)
        return sum / 2;
    }
}
