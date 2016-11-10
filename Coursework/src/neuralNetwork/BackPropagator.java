package neuralNetwork;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class BackPropagator {

    private Perceptron neuralNetwork;
    private double learningRate;
    private double momentum;
    private double characteristicTime;

    public BackPropagator(Perceptron neuralNetwork, double learningRate, double momentum, double characteristicTime) 
    {
        this.neuralNetwork = neuralNetwork;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.characteristicTime = characteristicTime;
    }

    public void train(TrainerGenerator generator, double errorThreshold) 
    {

        double error = 0.0;
        double sum = 0.0;
        double average = 25;
        int epoch = 1;
        int samples = 25;
        double[] errors = new double[samples];

        do {
            Trainer trainingData = generator.getTrainer();
            error = backPropagate(trainingData.getInputs(), trainingData.getOutputs());

            sum -= errors[epoch % samples];
            errors[epoch % samples] = error;
            sum += errors[epoch % samples];

            if(epoch > samples) {
                average = sum / samples;
            }

            System.out.println("Error for epoch " + epoch + ": " + error + ". Average: " + average);
            epoch++;
        } while(average > errorThreshold);
    }

    public double backPropagate(double[][] inputs, double[][] expectedOutputs) 
    {

        double error = 0;

        Map<Synapse, Double> synapseNeuronDeltaMap = new HashMap<Synapse, Double>();

        for (int i = 0; i < inputs.length; i++) {

            double[] input = inputs[i];
            double[] expectedOutput = expectedOutputs[i];

            List<NeuralLayer> layers = neuralNetwork.getNeuralLayers();

            neuralNetwork.setInputs(input);
            double[] output = neuralNetwork.getOutput();
            
            //First step of the backpropagation algorithm. Backpropagate errors from the output layer all the way up
            //to the first hidden layer
            for (int j = layers.size() - 1; j > 0; j--) {
                NeuralLayer layer = layers.get(j);

                for (int k = 0; k < layer.getNeurons().size(); k++) {
                    Neuron neuron = layer.getNeurons().get(k);
                    double neuronError = 0;

                    if (layer.IsOutputNeuralLayer()) {
                        //the order of output and expected determines the sign of the delta. if we have output - expected, we subtract the delta
                        //if we have expected - output we add the delta.
             // modified --------------------------------------------------------------------------------
                    	for(int l = 0; l < layer.getNeurons().size(); l++) 
                    	{
                    			neuronError = neuron.getDerivative() * (output[k] - expectedOutput[k]);
                    	} 
                    	
            // fin modified ------------------------------------------------------------------------------
                    } else {
                        neuronError = neuron.getDerivative();

                        double sum = 0;
                        List<Neuron> downstreamNeurons = layer.getNextNeuralLayer().getNeurons();
                        for (Neuron downstreamNeuron : downstreamNeurons) {

                            int l = 0;
                            boolean found = false;
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
            for(int j = layers.size() - 1; j > 0; j--) {
                NeuralLayer layer = layers.get(j);

                for(Neuron neuron : layer.getNeurons()) {

                    for(Synapse synapse : neuron.getInputs()) {
                        double delta = learningRate * neuron.getError() * synapse.getSourceNeuron().getOutput(); 

                        if(synapseNeuronDeltaMap.get(synapse) != null) {
                            double previousDelta = synapseNeuronDeltaMap.get(synapse);
                            delta += momentum * previousDelta;
                        }

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


    public double error(double[] actual, double[] expected) 
    {

        if (actual.length != expected.length) {
            throw new IllegalArgumentException("The lengths of the actual and expected value arrays must be equal");
        }

        double sum = 0;

        for (int i = 0; i < expected.length; i++) {
            sum += Math.pow(expected[i] - actual[i], 2);
        }

        return sum / 2;
    }
}