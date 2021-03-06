#pragma once
#include "Layer.h"
#include "NNData.h"

struct ExampleError
{
	double leftError;
	double rightError;
};

class NeuralNetwork
{
public:
	NeuralNetwork();
	NeuralNetwork(double, double, double, int);
	pair<double, double> train(vector<TrainingExample>&, int, int);
	double testNetwork(vector<TrainingExample>&, int, int);
	vector<vector<double>> getHiddenWeights();
	vector<vector<double>> getOutputWeights();
	void newEpoch();
private:
	int numberOfHiddenNeurons;
	double momentum, learningRate, lambda;
	ExampleError feedForward(TrainingExample&);
	void backPropagate(TrainingExample&, ExampleError&);
	double calculateAverageRmse(vector<ExampleError>&);
	Layer layers[2];
};