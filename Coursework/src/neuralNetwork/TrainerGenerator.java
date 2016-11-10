package neuralNetwork;

//This interface is used to specify to properties a trainer must have
public interface TrainerGenerator {
    Trainer getTrainer();
}