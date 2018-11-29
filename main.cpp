#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "NNData.h"
#include "NeuralNetwork.h"

using namespace std;

bool stoppingCriterion(vector<double>& errors)
{
	int lastErrors = 6;
	if (errors.empty() || errors.size() < lastErrors)
		return false;
	/*if (errors[errors.size() - 1] - errors[errors.size() - lastErrors] < 0.0005)
		return true;*/
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

void trainNetwork()
{
	ifstream csvFile("cleanNormalisedTest.csv"); // Input file
	ofstream weightOutput, errorStats; // Files for weights and stats
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
	csvFile.close();

	// Prepare data for training
	trainingData.randomiseAll();
	trainingData.divideData(0.70);
	trainingData.divideValidationData(0.15);

	// Set parameters
	double learningRate = 0.1, momentum = 0.01, lambda = 0.7;
	int hiddenNeurons = 4;

	//Initialize network
	NeuralNetwork neuralNetwork(learningRate, momentum, lambda, hiddenNeurons);

	int epoch = 0;
	vector<double> validationErrors;
	pair<double, double> errorPair; // Temp variable to receive training and validation errors, in that order, from NeuralNetwork.train()
	vector<vector<vector<double>>> hiddenWeightsForEpoch; // Stores hidden weights for each epoch
	vector<vector<vector<double>>> outputWeightsForEpoch; // Stores output weights for each epoch

														  // Generate error stats file name according to parameters selected
	string errorFileName = "learningR" + to_string(learningRate) + "momentum" + to_string(momentum) + "lambda" + to_string(lambda) + "hidden" + to_string(hiddenNeurons) + ".csv";
	errorStats.open(errorFileName.c_str());

	// Main training loop
	while (!stoppingCriterion(validationErrors) && epoch < 506)
	{
		errorPair = neuralNetwork.train(*(trainingData.getExamples()), trainingData.getNumberOfTrainingExamples(), trainingData.getNumberOfValidationExamples());
		validationErrors.push_back(errorPair.second);
		hiddenWeightsForEpoch.push_back(neuralNetwork.getHiddenWeights());
		outputWeightsForEpoch.push_back(neuralNetwork.getOutputWeights());
		cout << "Epoch " << epoch << ": " << "Training error: " << errorPair.first << " Validation error: " << errorPair.second << endl;
		errorStats << errorPair.first << ',' << errorPair.second << '\n';
		epoch++;
	}
	errorStats.close();

	neuralNetwork.testNetwork(*(trainingData.getExamples()), trainingData.getNumberOfTrainingExamples(), trainingData.getNumberOfValidationExamples());

	weightOutput.open("weights.txt");

	const int selectedEpoch = epoch - 6;

	// Weight output contains dimensions of weight matrix separated by a space in a line followed by matrix with elements separated by spaces line by line
	weightOutput << hiddenWeightsForEpoch[selectedEpoch].size() << ' ' << hiddenWeightsForEpoch[selectedEpoch][0].size();

	// Print weight results to screen and file
	cout << endl << "Hidden weights for selected epoch no. " << selectedEpoch << endl << endl;

	for (int i = 0; i < hiddenWeightsForEpoch[selectedEpoch].size(); i++)
	{
		for (int j = 0; j < hiddenWeightsForEpoch[selectedEpoch][i].size(); j++)
		{
			cout << hiddenWeightsForEpoch[selectedEpoch][i][j] << " ";
			weightOutput << hiddenWeightsForEpoch[selectedEpoch][i][j] << " ";
		}
		cout << endl;
		weightOutput << "\n";
	}

	weightOutput << outputWeightsForEpoch[selectedEpoch].size() << ' ' << outputWeightsForEpoch[selectedEpoch][0].size();

	cout << endl << "Output weights for selected epoch no. " << selectedEpoch << endl << endl;

	for (int i = 0; i < outputWeightsForEpoch[selectedEpoch].size(); i++)
	{
		for (int j = 0; j < outputWeightsForEpoch[selectedEpoch][i].size(); j++)
		{
			cout << outputWeightsForEpoch[selectedEpoch][i][j] << " ";
			weightOutput << outputWeightsForEpoch[selectedEpoch][i][j] << " ";
		}
		cout << endl;
		weightOutput << "\n";
	}
}

void runRobot()
{
	
}

int main(int argc, char* argv[])
{
	char trainOrRun;
	cout << "Train network or run it in robot? (T/R): ";
	cin >> trainOrRun;
	if (trainOrRun == 'T')
		trainNetwork();
	else
		runRobot();
	
	

	return 0;
}