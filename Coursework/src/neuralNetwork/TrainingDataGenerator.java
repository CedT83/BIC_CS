package neuralNetwork;

import java.util.Random;


public class TrainingDataGenerator implements TrainerGenerator {

	//We define here the input data used for training
    double[][] inputs = {{0.2, 0.7}, {0.4, 0.4}, {0.1, 0.2}, {0.8, 0.8}};
    //We define here the expected output for the inputs defined above
    double[][] outputs = {{0.53}, {0.32}, {0.17}, {1.28}};
    int[] inputIndices = new int[inputs.length];
    
    public Trainer getTrainer() {
        double[][] randomizedInputs = new double[inputs.length][inputs[0].length];
        double[][] randomizedOutputs = new double[outputs.length][outputs[0].length];

        //To be sure the data used for training has a random order
        for(int i=0; i<inputs.length; i++){
        	inputIndices[i] = i;
       }
        inputIndices = shuffle(inputIndices);

        for(int i = 0; i < inputIndices.length; i++) {
            randomizedInputs[i] = inputs[inputIndices[i]];
            randomizedOutputs[i] = outputs[inputIndices[i]];
        }

        return new Trainer(randomizedInputs, randomizedOutputs);
    }

    //This function takes an array and return an array with random ordered values from paramter
    private int[] shuffle(int[] array) {

        Random random = new Random();
        for(int i = array.length - 1; i > 0; i--) {

            int index = random.nextInt(i + 1);

            int temp = array[i];
            array[i] = array[index];
            array[index] = temp;
        }
        return array;
    }
}
