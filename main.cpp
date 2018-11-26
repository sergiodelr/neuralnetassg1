#include <iostream>
#include <vector>
#include <fstream>
#include "NNData.h"
#include "NeuralNetwork.h"

using namespace std;

bool stoppingCriterion(vector<double>& errors)
{
	int lastErrors = 6;
	if (errors.empty() || errors.size() < lastErrors + 1)
		return false;
	int i = 0;
	bool stop = true;
	while (stop && i < lastErrors)
	{
		if (errors[errors.size() - i - 1] < errors[errors.size() - i - 2])
			stop = false;
		i++;
	}

	return stop;
}

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
	vector<double> validationErrors;
	pair<double, double> errorPair;
	int epoch = 1;
	while (!stoppingCriterion(validationErrors))
	{
		errorPair = neuralNetwork.train(*(trainingData.getExamples()), trainingData.getNumberOfTrainingExamples, trainingData.getNumberOfValidationExamples);
		validationErrors.push_back(errorPair.second);
		cout << "Epoch " << epoch << ": " << "Training error: " << errorPair.first << " Validation error: " << errorPair.second << endl;
	}
	
	return 0;
}
