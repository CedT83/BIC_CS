package neuralNetwork;

public class Program {

	public static void main(String[] args) {

		//We create our perceptron here and define its architecture
        Perceptron perceptron = createUntrainedPerceptron();
        //Creates the trainer for the perceptron using training data
        TrainerGenerator TrainingDataGenerator = new TrainingDataGenerator();

        //crates the packpropagator object with the perceptron and its characteristics
        BackPropagator backpropagator = new BackPropagator(perceptron, 0.8, 0.9, 0);
        //We define here the desired trainer and the minimal error
        //And we train until the error is reached
        backpropagator.train(TrainingDataGenerator, 0.0001);

        //A line to show the training is done and can proceed to the tests
        System.out.println("Testing untrained neural network");

        //We choose the test values, and print the result given by the neural network
        perceptron.setInputs(new double[]{0, 0});
        System.out.println("0 - 0: " + (perceptron.getOutput()[0]));

        //We repeat with other values
        perceptron.setInputs(new double[]{.17, 1});
        System.out.println("0.17 - 0.88: " + (perceptron.getOutput()[0]));

        perceptron.setInputs(new double[]{0.02, 0.44});
        System.out.println("0.02 - 0.44: " + (perceptron.getOutput()[0]));

        perceptron.setInputs(new double[]{1, 1});
        System.out.println("1 - 1: " + (perceptron.getOutput()[0]) + "\n");
    }
    
    private static Perceptron createUntrainedPerceptron() {
        Perceptron NeuralNetwork = new Perceptron("Untrained Network");

        Neuron inputBias = new Neuron(new SigmoidActivationFunction());
        inputBias.setOutput(1);
        NeuralLayer inputLayer = new NeuralLayer(null, inputBias);

        Neuron A = new Neuron(new SigmoidActivationFunction());
        A.setOutput(0);

        Neuron B = new Neuron(new SigmoidActivationFunction());
        B.setOutput(0);
        

        inputLayer.AddNeuron(A);
        inputLayer.AddNeuron(B);

        

        ////////////////// 1st hidden layer
        Neuron bias = new Neuron(new SigmoidActivationFunction());
        bias.setOutput(1);
        NeuralLayer hiddenLayer = new NeuralLayer(inputLayer, bias);

        Neuron hiddenA = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenB = new Neuron(new SigmoidActivationFunction());
//        Neuron hiddenC = new Neuron(new SigmoidActivationFunction());
//        Neuron hiddenD = new Neuron(new SigmoidActivationFunction());



        hiddenLayer.AddNeuron(hiddenA);
        hiddenLayer.AddNeuron(hiddenB);
//        hiddenLayer.AddNeuron(hiddenC);
//        hiddenLayer.AddNeuron(hiddenD);


     
        //////////////////// 2nd hidden layer
        bias.setOutput(1);
        NeuralLayer hiddenLayer2 = new NeuralLayer(hiddenLayer, bias);

        Neuron hiddenA2 = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenB2 = new Neuron(new SigmoidActivationFunction());
//        Neuron hiddenC2 = new Neuron(new SigmoidActivationFunction());
//        Neuron hiddenD2 = new Neuron(new SigmoidActivationFunction());




        hiddenLayer.AddNeuron(hiddenA2);
        hiddenLayer.AddNeuron(hiddenB2); 
//        hiddenLayer.AddNeuron(hiddenC2); 
//        hiddenLayer.AddNeuron(hiddenD2); 

        
        NeuralLayer outputLayer = new NeuralLayer(hiddenLayer2);
        Neuron Neuron = new Neuron(new SigmoidActivationFunction());
        outputLayer.AddNeuron(Neuron);

        NeuralNetwork.addNeuralLayer(inputLayer);
        NeuralNetwork.addNeuralLayer(hiddenLayer);
        NeuralNetwork.addNeuralLayer(hiddenLayer);
        NeuralNetwork.addNeuralLayer(outputLayer);

        return NeuralNetwork;
    }
}
