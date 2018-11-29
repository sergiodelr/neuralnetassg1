#include "Layer.h"
#include <ctime>

using namespace std;

Layer::Layer()
{
}

// Constructor
Layer::Layer(int nodesInPreviousLayer, int nodesCurrentLayer, double lambda, double learningRate, double momentum)
{
	size = nodesCurrentLayer;
	weights = vector<vector<double>>(size, vector<double>(nodesInPreviousLayer + 1, 0));
	for (int i = 0; i < size; i++)
		for (int j = 0; j < nodesInPreviousLayer + 1; j++)
			weights[i][j] = (double)rand() / RAND_MAX;
	previousWeights = weights;
	activations = vector<double>(size, 0);
	localGradients = vector<double>(size, 0);
	this->lambda = lambda;
	this->learningRate = learningRate;
	this->momentum = momentum;
	previousWeightDeltas = vector<vector<double>>(size, vector<double>(nodesInPreviousLayer + 1, 0));
}

// Performs a forward pass of the data from the previous layer and computes activations
// param previousActivations - Activations from previous layer. Has to be the same size as of nodesInPreviousLayer.
void Layer::feedForward(vector<double>* previousActivations)
{
	if (previousActivations->size() == weights[0].size() - 1)
	{
		// Perform dot product between previous activations and weights
		for (int i = 0; i < size; i++)
		{
			activations[i] = 0;

			for (int j = 0; j < weights[0].size() - 1; j++)
				activations[i] += weights[i][j] * (*previousActivations)[j];
			// Add bias
			activations[i] += weights[i][weights[0].size() - 1];
		}
		// Perform sigmoid function
		for (int i = 0; i < activations.size(); i++)
			activations[i] = sigmoid(activations[i]);
	}
}

int Layer::getSize()
{
	return size;
}

vector<double>* Layer::getActivations()
{
	return &activations;
}

vector<vector<double>>* Layer::getPreviousWeights()
{
	return &previousWeights;
}

vector<double>* Layer::getLocalGradients()
{
	return &localGradients;
}

void Layer::backPropagateOutputLayer(double leftError, double rightError, vector<double>* previousActivations)
{
	localGradients[0] = lambda * activations[0] * (1 - activations[0]) * leftError;
	localGradients[1] = lambda * activations[1] * (1 - activations[1]) * rightError;

	vector<vector<double>> currentWeightDeltas(size, vector<double>(weights[0].size(), 0));

	previousWeights = weights;

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < weights[i].size() - 1; j++)
		{
			// Compute weight deltas
			currentWeightDeltas[i][j] = learningRate * localGradients[i] * (*previousActivations)[j] + momentum * previousWeightDeltas[i][j];
		}
		// Compute delta for bias
		currentWeightDeltas[i][weights[i].size() - 1] = learningRate * localGradients[i] + momentum * previousWeightDeltas[i][weights[i].size() - 1];
	}

	// Update weights
	for (int i = 0; i < size; i++)
		for (int j = 0; j < weights[i].size(); j++)
			weights[i][j] += currentWeightDeltas[i][j];

	previousWeightDeltas = currentWeightDeltas;
}

void Layer::backPropagateHiddenLayer(vector<double>* localGradientsOutput, vector<vector<double>>* previousWeightsOutput, vector<double>& example)
{
	double sumTerm;
	for (int i = 0; i < size; i++)
	{
		sumTerm = 0;
		for (int j = 0; j < previousWeightsOutput->size(); j++)
		{
			sumTerm += (*previousWeightsOutput)[j][i] * (*localGradientsOutput)[j];
		}
		localGradients[i] = lambda * activations[i] * (1 - activations[i]) * sumTerm;
	}

	vector<vector<double>> currentWeightDeltas(size, vector<double>(weights[0].size(), 0));

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < weights[i].size() - 1; j++)
		{
			// Compute weight deltas
			currentWeightDeltas[i][j] = learningRate * localGradients[i] * example[j] + momentum * previousWeightDeltas[i][j];
		}
		// Compute delta for bias
		currentWeightDeltas[i][weights[i].size() - 1] = learningRate * localGradients[i] + momentum * previousWeightDeltas[i][weights[i].size() - 1];
	}

	// Update weights
	for (int i = 0; i < size; i++)
		for (int j = 0; j < weights[i].size(); j++)
			weights[i][j] += currentWeightDeltas[i][j];

	previousWeightDeltas = currentWeightDeltas;
}

void Layer::reset()
{
	for (int i = 0; i < previousWeights.size(); i++)
	{
		for (int j = 0; j < previousWeights[0].size(); j++)
		{
			previousWeights[i][j] = 0;
			previousWeightDeltas[i][j] = 0;
		}
	}
}

vector<vector<double>> Layer::getWeights()
{
	return weights;
}

double Layer::sigmoid(double x)
{
	return 1.0 / (1 + exp(-lambda * x));
}