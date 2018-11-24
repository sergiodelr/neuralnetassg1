#include "NeuralNetwork.h"


// Default constructor
NeuralNetwork::NeuralNetwork()
{
	this->learningRate = 0;
	this->momentum = 0;
	this->lambda = 0;
	this->numberOfHiddenNeurons = 0;
}

// Constructor
NeuralNetwork::NeuralNetwork(double learningRate, double momentum, double lambda, int numberOfHiddenNeurons)
{
	this->learningRate = learningRate;
	this->momentum = momentum;
	this->lambda = lambda;
	this->numberOfHiddenNeurons = numberOfHiddenNeurons;
	layers[0] = Layer(2, numberOfHiddenNeurons, lambda, learningRate, momentum);
	layers[1] = Layer(numberOfHiddenNeurons, 2, lambda, learningRate, momentum);
}

// Trains the Neural Network with the data provided
// params examples - vector of training examples
//		  numberOfTrainingExamples - number of examples that are going to be used for training within the vector
//		  numberOfValidationExamples - number of examples that are going to be used for validation within the vector
pair<double, double> NeuralNetwork::train(vector<TrainingExample>& examples, int numberOfTrainingExamples, int numberOfValidationExamples)
{
	vector<ExampleError> exampleErrorsTraining, exampleErrorsValidation;
	for (int i = 0; i < numberOfTrainingExamples; i++)
	{
		exampleErrorsTraining.push_back(feedForward(examples[i]));
		backPropagate(examples[i], exampleErrorsTraining[i]);
	}

	for (int i = numberOfTrainingExamples; i < numberOfTrainingExamples + numberOfValidationExamples; i++)
	{
		exampleErrorsValidation.push_back(feedForward(examples[i]));
	}

	
}

// Makes a feedforward pass of the training example through the network and returns the error
// param example - the training example to be passed through the network
// return - error between expected and actual output
ExampleError NeuralNetwork::feedForward(TrainingExample& example)
{
	vector<double> inputVector;
	inputVector.push_back(example.frontSensor);
	inputVector.push_back(example.backSensor);

	layers[0].feedForward(&inputVector);
	layers[1].feedForward(layers[0].getActivations());

	ExampleError exampleError;

	exampleError.leftError = example.leftWheel - (*layers[1].getActivations())[0];
	exampleError.rightError = example.rightWheel - (*layers[1].getActivations())[1];

	return exampleError;
}

void NeuralNetwork::backPropagate(TrainingExample& example, ExampleError& error)
{
	vector<double> tempError;
	tempError.push_back(error.leftError);
	tempError.push_back(error.rightError);

	layers[1].backPropagateOutputLayer(error.leftError, error.rightError, layers[0].getActivations());
	layers[0].backPropagateHiddenLayer(layers[1].getLocalGradients, layers[1].getPreviousWeights, tempError);
}

double NeuralNetwork::calculateRmse(vector<TrainingExample>&)
{
	return 0.0;
}

