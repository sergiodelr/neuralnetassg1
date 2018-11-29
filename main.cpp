#include <iostream>
#include <vector>
#include <fstream>
#include "NNData.h"
#include "NeuralNetwork.h"

using namespace std;

bool stoppingCriterion(vector<double>& errors)
{
	int lastErrors = 6;
	if (errors.empty() || errors.size() < lastErrors)
		return false;
	if (errors[errors.size() - 1] - errors[errors.size() - lastErrors] < 0.0005)
		return true;
	int i = 1;
	bool stop = true;
	while (stop && i < lastErrors)
	{
		if (errors[errors.size() - i] < errors[errors.size() - i - 1])
			stop = false;
		i++;
	}

	return stop;
}

int main(int argc, char* argv[])
{
	ifstream csvFile("C:\\Users\\sgo_a\\Documents\\nn\\NeuralNetworksAssg1\\NeuralNetworksAssg1\\x64\\Debug\\cleanNormalisedTest.csv");
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

	double learningRate = 0.1, momentum = 0.01, lambda = 0.7;
	int hiddenNeurons = 4;
	NeuralNetwork neuralNetwork(learningRate, momentum, lambda, hiddenNeurons);
	vector<double> validationErrors;
	pair<double, double> errorPair;

	int epoch = 0;
	vector<vector<vector<double>>> hiddenWeightsForEpoch;
	vector<vector<vector<double>>> outputWeightsForEpoch;

	while (!stoppingCriterion(validationErrors))
	{
		errorPair = neuralNetwork.train(*(trainingData.getExamples()), trainingData.getNumberOfTrainingExamples(), trainingData.getNumberOfValidationExamples());
		validationErrors.push_back(errorPair.second);
		hiddenWeightsForEpoch.push_back(neuralNetwork.getHiddenWeights());
		outputWeightsForEpoch.push_back(neuralNetwork.getOutputWeights());
		cout << "Epoch " << epoch << ": " << "Training error: " << errorPair.first << " Validation error: " << errorPair.second << endl;
		
		epoch++;
	}

	int selectedEpoch = epoch - 6;

	cout << endl << "Hidden weights for selected epoch no. " << selectedEpoch << endl << endl;

	for (int i = 0; i < hiddenWeightsForEpoch[selectedEpoch].size(); i++)
	{
		for (int j = 0; j < hiddenWeightsForEpoch[selectedEpoch][i].size(); j++)
		{
			cout << hiddenWeightsForEpoch[selectedEpoch][i][j] << " ";
		}
		cout << endl;
	}

	cout << endl << "Output weights for selected epoch no. " << selectedEpoch << endl << endl;

	for (int i = 0; i < outputWeightsForEpoch[selectedEpoch].size(); i++)
	{
		for (int j = 0; j < outputWeightsForEpoch[selectedEpoch][i].size(); j++)
		{
			cout << outputWeightsForEpoch[selectedEpoch][i][j] << " ";
		}
		cout << endl;
	}

	return 0;
}