#include "NNData.h"

#include <algorithm>


NNData::NNData()
{
	numberOfTrainingExamples = 0;
	numberOfValidationExamples = 0;
}


// Adds a training example
void NNData::addExample(double frontSensor, double backSensor, double leftWheel, double rightWheel)
{
	TrainingExample trainingExample;
	trainingExample.frontSensor = frontSensor;
	trainingExample.backSensor = backSensor;
	trainingExample.leftWheel = leftWheel;
	trainingExample.rightWheel = rightWheel;
	dataExamples.push_back(trainingExample);
}

// Divides the data into training and testing according to the percentage stated
void NNData::divideData(double trainingPercentage)
{
	numberOfTrainingExamples = dataExamples.size() * trainingPercentage;
}

// Divides the remaining data into test and validation according to the validation percentage of the TOTAL data
// stated
void NNData::divideValidationData(double validationPercentage)
{
	numberOfValidationExamples = dataExamples.size() * validationPercentage;
}

void NNData::randomiseAll()
{
	random_shuffle(dataExamples.begin(), dataExamples.end());
}

void NNData::randomiseTraining()
{
	random_shuffle(dataExamples.begin(), dataExamples.begin() + numberOfTrainingExamples + numberOfValidationExamples);
}

int NNData::getNumberOfTrainingExamples()
{
	return numberOfTrainingExamples;
}

int NNData::getNumberOfValidationExamples()
{
	return numberOfValidationExamples;
}

vector<TrainingExample>* NNData::getExamples()
{
	return &dataExamples;
}