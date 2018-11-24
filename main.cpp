#include <iostream>
#include <vector>
#include <fstream>
#include "NNData.h"
#include "NeuralNetwork.h"

using namespace std;

int main(int argc, char* argv[])
{
	ifstream csvFile("train.csv");
	NNData trainingData;

	// Read data from csv 
	double frontSensor, backSensor, leftWheelVelocity, rightWheelVelocity;
	char comma;

	while (csvFile >> frontSensor)
	{
		csvFile >> comma >> backSensor;
		csvFile >> comma >> leftWheelVelocity;
		csvFile >> comma >> rightWheelVelocity;
		trainingData.addExample(frontSensor, backSensor, leftWheelVelocity, rightWheelVelocity);
	}

	// Prepare data for training
	trainingData.randomiseAll();
	trainingData.divideData(0.70);
	trainingData.divideValidationData(0.15);
	
	double learningRate = 0.3, momentum = 0.01, hiddenNeurons = 4, lambda = 0.7;
	NeuralNetwork neuralNetwork(learningRate, momentum, lambda, hiddenNeurons);
	neuralNetwork.train(*(trainingData.getExamples()), trainingData.getNumberOfTrainingExamples, trainingData.getNumberOfValidationExamples);

}
