package neuralNetwork;

import java.util.Random;


public class TrainingDataGenerator implements TrainerGenerator {

    double[][] inputs = {{0.2, 0.7}, {0.4, 0.4}, {0.1, 0.2}, {0.8, 0.8}};
    double[][] outputs = {{0.53}, {0.32}, {0.17}, {1.28}};
    int[] inputIndices = {0, 1, 2, 3};

    public Trainer getTrainer() {
        double[][] randomizedInputs = new double[4][2];
        double[][] randomizedOutputs = new double[4][1];

        inputIndices = shuffle(inputIndices);

        for(int i = 0; i < inputIndices.length; i++) {
            randomizedInputs[i] = inputs[inputIndices[i]];
            randomizedOutputs[i] = outputs[inputIndices[i]];
        }

//        return new TrainingData(inputs, outputs);
        return new Trainer(randomizedInputs, randomizedOutputs);
    }

    private int[] shuffle(int[] array) {

        Random random = new Random();
        for(int i = array.length - 1; i > 0; i--) {

            int index = random.nextInt(i + 1);

            int temp = array[i];
            array[i] = array[index];
            array[index] = temp;
        }

        return array;
//        return new int[]{2, 1, 3, 0};
    }
}
