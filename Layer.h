#pragma once
#include <vector>

using namespace std;

class Layer
{
public:
	Layer();
	Layer(int, int, double, double, double);
	void feedForward(vector<double>*);
	int getSize();
	vector<double>* getActivations();
	vector<vector<double>>* getPreviousWeights();
	vector<double>* getLocalGradients();
	void backPropagateOutputLayer(double, double, vector<double>*);
	void backPropagateHiddenLayer(vector<double>*, vector<vector<double>>*, vector<double>&);
	void reset();
	vector<vector<double>> getWeights();
private:
	double sigmoid(double);
	vector<vector<double>> weights;
	vector<vector<double>> previousWeights;
	vector<vector<double>> previousWeightDeltas;
	vector<double> localGradients;
	vector<double> activations;
	double lambda, learningRate, momentum;
	int size;
};