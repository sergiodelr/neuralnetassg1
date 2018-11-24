#pragma once
#include <utility>
#include <vector>

using namespace std;

struct TrainingExample
{
	double frontSensor;
	double backSensor;
	double leftWheel;
	double rightWheel;
};

class NNData
{
public:
	NNData();
	void addExample(double, double, double, double);
	void divideData(double);
	void divideValidationData(double);
	void randomiseAll();
	void randomiseTraining();
	int getNumberOfTrainingExamples();
	int getNumberOfValidationExamples();
	vector<TrainingExample>* getExamples();
private:
	vector<TrainingExample> dataExamples;
	int numberOfTrainingExamples;
	int numberOfValidationExamples;
};
