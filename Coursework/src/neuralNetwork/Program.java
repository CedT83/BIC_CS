package neuralNetwork;

public class Program {

	public static void main(String[] args) {


        Perceptron untrained = createUntrainedXorPerceptron();
        TrainerGenerator xorTrainingDataGenerator = new TrainingDataGenerator();

        BackPropagator backpropagator = new BackPropagator(untrained, 0.8, 0.9, 0);
        backpropagator.train(xorTrainingDataGenerator, 0.0001);

        System.out.println("Testing trained XOR neural network");

        untrained.setInputs(new double[]{0, 0});
        System.out.println("0 - 0: " + (untrained.getOutput()[0]));

        untrained.setInputs(new double[]{0, 1});
        System.out.println("0 - 1: " + (untrained.getOutput()[0]));

        untrained.setInputs(new double[]{1, 0});
        System.out.println("1 - 0: " + (untrained.getOutput()[0]));

        untrained.setInputs(new double[]{1, 1});
        System.out.println("1 - 1: " + (untrained.getOutput()[0]) + "\n");

        untrained.persist();
    }

	
    private static Perceptron createXorPerceptron() {
        Perceptron xorPerceptron = new Perceptron("XOR Network");

        NeuralLayer inputNeuralLayer = new NeuralLayer(null);

        Neuron a = new Neuron(new SigmoidActivationFunction());
        a.setOutput(0);

        Neuron b = new Neuron(new SigmoidActivationFunction());
        b.setOutput(0);

        inputNeuralLayer.AddNeuron(a);
        inputNeuralLayer.AddNeuron(b);

        NeuralLayer hiddenNeuralLayer = new NeuralLayer(inputNeuralLayer);

        Neuron hiddenA = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenB = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenC = new Neuron(new SigmoidActivationFunction());

        hiddenNeuralLayer.AddNeuron(hiddenA);
        hiddenNeuralLayer.AddNeuron(hiddenB);
        hiddenNeuralLayer.AddNeuron(hiddenC);
        
        
        NeuralLayer hiddenNeuralLayer2 = new NeuralLayer(inputNeuralLayer);

        Neuron hiddenA2 = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenB2 = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenC2 = new Neuron(new SigmoidActivationFunction());

        hiddenNeuralLayer.AddNeuron(hiddenA2);
        hiddenNeuralLayer.AddNeuron(hiddenB2);
        hiddenNeuralLayer.AddNeuron(hiddenC2);

        NeuralLayer outputNeuralLayer = new NeuralLayer(hiddenNeuralLayer2);
        Neuron xorNeuron = new Neuron(new SigmoidActivationFunction());
        outputNeuralLayer.AddNeuron(xorNeuron);

        xorPerceptron.addNeuralLayer(inputNeuralLayer);
        xorPerceptron.addNeuralLayer(hiddenNeuralLayer);
        xorPerceptron.addNeuralLayer(outputNeuralLayer);

        return xorPerceptron;
    }
    
    private static Perceptron createUntrainedXorPerceptron() {
        Perceptron xorNeuralNetwork = new Perceptron("Trained XOR Network");

        Neuron inputBias = new Neuron(new SigmoidActivationFunction());
        inputBias.setOutput(1);
        NeuralLayer inputLayer = new NeuralLayer(null, inputBias);

        Neuron a = new Neuron(new SigmoidActivationFunction());
        a.setOutput(0);

        Neuron b = new Neuron(new SigmoidActivationFunction());
        b.setOutput(0);
        

        inputLayer.AddNeuron(a);
        inputLayer.AddNeuron(b);

        

////////////////// 1st hidden layer
        Neuron bias = new Neuron(new SigmoidActivationFunction());
        bias.setOutput(1);
        NeuralLayer hiddenLayer = new NeuralLayer(inputLayer, bias);

        Neuron hiddenA = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenB = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenC = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenD = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenE = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenF = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenG = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenH = new Neuron(new SigmoidActivationFunction());


        hiddenLayer.AddNeuron(hiddenA);
        hiddenLayer.AddNeuron(hiddenB);
        hiddenLayer.AddNeuron(hiddenC);
        hiddenLayer.AddNeuron(hiddenD);
        hiddenLayer.AddNeuron(hiddenE);
        hiddenLayer.AddNeuron(hiddenF);
        hiddenLayer.AddNeuron(hiddenG);
        hiddenLayer.AddNeuron(hiddenH); 


     
//////////////////// 2nd hidden layer
   /*               bias.setOutput(1);
        NeuralLayer hiddenLayer2 = new NeuralLayer(hiddenLayer, bias);

        Neuron hiddenA2 = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenB2 = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenC2 = new Neuron(new SigmoidActivationFunction());

        Neuron hiddenD2 = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenE2 = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenF2 = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenG2 = new Neuron(new SigmoidActivationFunction());
        Neuron hiddenH2 = new Neuron(new SigmoidActivationFunction());



        hiddenLayer.AddNeuron(hiddenA2);
        hiddenLayer.AddNeuron(hiddenB2); 
        hiddenLayer.AddNeuron(hiddenC2); 
        hiddenLayer.AddNeuron(hiddenD2); 
        hiddenLayer.AddNeuron(hiddenE2);
        hiddenLayer.AddNeuron(hiddenF2);
        hiddenLayer.AddNeuron(hiddenG2);
        hiddenLayer.AddNeuron(hiddenH2); 

////////////////////3rd hidden layer
/*bias.setOutput(1);
NeuralLayer hiddenLayer3 = new NeuralLayer(hiddenLayer2, bias);

Neuron hiddenA3 = new Neuron(new SigmoidActivationFunction());
Neuron hiddenB3 = new Neuron(new SigmoidActivationFunction());
Neuron hiddenC3 = new Neuron(new SigmoidActivationFunction());

Neuron hiddenD3 = new Neuron(new SigmoidActivationFunction());
Neuron hiddenE3 = new Neuron(new SigmoidActivationFunction());
Neuron hiddenF3 = new Neuron(new SigmoidActivationFunction());
Neuron hiddenG3 = new Neuron(new SigmoidActivationFunction());
Neuron hiddenH3 = new Neuron(new SigmoidActivationFunction());



hiddenLayer.AddNeuron(hiddenA3);
hiddenLayer.AddNeuron(hiddenB3); 
hiddenLayer.AddNeuron(hiddenC3); 
hiddenLayer.AddNeuron(hiddenD2); 
hiddenLayer.AddNeuron(hiddenE2);
hiddenLayer.AddNeuron(hiddenF2);
hiddenLayer.AddNeuron(hiddenG2);
hiddenLayer.AddNeuron(hiddenH2);  */
        
        NeuralLayer outputLayer = new NeuralLayer(hiddenLayer);
        Neuron xorNeuron = new Neuron(new SigmoidActivationFunction());
        outputLayer.AddNeuron(xorNeuron);

        xorNeuralNetwork.addNeuralLayer(inputLayer);
        xorNeuralNetwork.addNeuralLayer(hiddenLayer);
 //       xorNeuralNetwork.addNeuralLayer(hiddenLayer2);
    //    xorNeuralNetwork.addNeuralLayer(hiddenLayer3);
        xorNeuralNetwork.addNeuralLayer(outputLayer);

        return xorNeuralNetwork;
    }

}
