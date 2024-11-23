#pragma once


#include "multy-layer-perceptron.h"


MLP::MLP()
{
}

MLP::~MLP()
{
}



std::vector<double> MLP::Foward(std::vector<double> input)
{
	std::vector<double> layerOutput  =  input;

	for (auto& layer : _layers) {
		layerOutput  =  layer.CalculateLayerOutputs( layerOutput );
	}

	return std::vector<double>( layerOutput.begin()+1, layerOutput.end() );          // return the predicted output
}



std::vector<double> MLP::Backward(std::vector<double> predictedValues, std::vector<double> correctValues)
{
	CalculateError(predictedValues, correctValues);

	int layerIndex  =  _layers.size() - 1;
	Eigen::MatrixXd dLoss_dInput;


	//--- update last layer
	dLoss_dInput = _layers[layerIndex--].UpdateLastLayerWeight(predictedValues, correctValues);


	//--- update hidden layers
	for (layerIndex; layerIndex >= 0; layerIndex--) {
		dLoss_dInput = _layers[layerIndex].UpdateHiddenLayerWeight( dLoss_dInput );
	}

	return  Utils::FlatMatrix(dLoss_dInput);
}


std::vector<double> MLP::Backward(std::vector<double> lossGradientWithRespectToOutput)
{
	int layerIndex  =  _layers.size() - 1;
	Eigen::MatrixXd dLoss_dInput  =  Utils::ReshapeMatrix(lossGradientWithRespectToOutput, lossGradientWithRespectToOutput.size(), 1);


	//--- update last layer
	dLoss_dInput = _layers[layerIndex--].UpdateHiddenLayerWeight( dLoss_dInput );


	//--- update hidden layers
	for (layerIndex; layerIndex >= 0; layerIndex--) {
		dLoss_dInput = _layers[layerIndex].UpdateHiddenLayerWeight(dLoss_dInput);
	}

	return  Utils::FlatMatrix(dLoss_dInput);
}


Eigen::MatrixXd MLP::Foward(Eigen::MatrixXd& input)
{
	Eigen::MatrixXd layerOutput  =  input;

	for (auto& layer : _layers) {
		layerOutput  =  layer.CalculateLayerOutputs(layerOutput);
	}

	size_t blockRows = layerOutput.rows() - 1;
	size_t blockCols = layerOutput.cols();
	return layerOutput.block(1, 0, blockRows, blockCols);   // igonre the bias output    // other neural network
}




Eigen::MatrixXd MLP::Backward(Eigen::MatrixXd& dL_dActivation)
{
	assert(dL_dActivation.cols()==1  &&  "[ERROR]: dL_dActivation must be a vertical vector [n x 1]!!!");

	Eigen::MatrixXd dLoss_dInput  =  dL_dActivation;
	int layerIndex  =  _layers.size() - 1;


	//--- update last layer
	dLoss_dInput = _layers[layerIndex--].UpdateHiddenLayerWeight(dLoss_dInput);


	//--- update hidden layers
	for (layerIndex; layerIndex >= 0; layerIndex--) {
		dLoss_dInput = _layers[layerIndex].UpdateHiddenLayerWeight(dLoss_dInput);
	}

	return  dLoss_dInput;    // [dLoss_dInput] = [n x 1]
}




void MLP::Training(std::vector<MLPTrainigData> trainigSet, std::function<void(void)> callback)
{
	bool keepGoing  =  true;
	size_t epoch  =  0;
	size_t trainingSetSize  =  trainigSet.size();

	Layer& lastLayer  =  _layers[_layers.size()-1];
	ILostFunction* loss = lastLayer._lostFunction;

	while (keepGoing) {

		for (size_t i = 0; i < trainingSetSize; i++) {
			std::vector<double> inputVec  =  trainigSet[i].INPUT;
			std::vector<double> label  =  trainigSet[i].LABEL;

			inputVec.insert(inputVec.begin(), 1.0);

			Eigen::MatrixXd input  =  Utils::ReshapeMatrix(inputVec, inputVec.size(), 1); 

			Eigen::MatrixXd predictedOutput  =  Foward(input);
			auto output  =  Utils::FlatMatrix(predictedOutput);

			// changed backward ---------------------------------------------------
			Eigen::MatrixXd dL_dActivation = Eigen::MatrixXd(predictedOutput.size(), 1);
			for (size_t i = 0; i < output.size(); i++) {
				dL_dActivation(i,0) = loss->df( output[i] , label[i]);
			}

			Eigen::MatrixXd dL_dInput = Backward( dL_dActivation );
			// END changed backward---------------------------------------------------
		}

		callback();

		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(trainigSet.begin(), trainigSet.end(), g);
		

		_error = _error / (double)trainingSetSize;
		ChangeLearningRate(epoch, _error);
		_error = 0.0;


		epoch++;
		if (epoch > _maxEpochs) {  keepGoing = false;  }
	}
}


void MLP::Training(std::vector<MLP_DATA> trainigSet, std::function<void(void)> callback)
{
	std::vector<MLPTrainigData> _trainingSet;

	for (auto data : trainigSet) {
		std::vector<double> label = ParseLabelToVector( data.labelIndex );
		_trainingSet.push_back({ data.input, label });
	}

	std::cout << "\nstart training:\n\n";
	Training(_trainingSet, callback);

	BuildJson();
}



std::vector<double> MLP::Classify(std::vector<double> input)
{
	input.insert(input.begin(), 1.0);
	return Foward(input);
}

size_t MLP::Classify(std::vector<double> input, std::function<size_t(std::vector<double>)> ParseOutputToLabel)
{
	input.insert(input.begin(), 1.0);
	std::vector<double> givemOutput  =  Foward( input );
	size_t label  =  ParseOutputToLabel( givemOutput );
	return label;
}

void MLP::Classify(std::vector<std::vector<double>> inputs, std::function<void(std::vector<double>)> CallBack)
{
	for (auto& input : inputs) {
		input.insert(input.begin(), 1.0);
		std::vector<double> givemOutput  =  Foward(input);
		CallBack(givemOutput);
	}
}

void MLP::Classify(std::vector<MLP_DATA> inputSet, std::function<void(std::vector<double>)> CallBack)
{
	for (auto& inputData : inputSet) {
		inputData.input.insert(inputData.input.begin(), 1.0);
		std::vector<double> givemOutput  =  Foward(inputData.input);
		CallBack(givemOutput);
	}
}



void MLP::ChangeLearningRate(size_t epoch, double error)
{
	for (auto& layer : _layers) {
		UpdateLeraningRate(epoch, error, layer._learningRate);
		//layer.Set<Layer::Attribute::LEARNING_RATE, double>(layer._learningRate);
	}
}


void MLP::CalculateError(std::vector<double> predictedValues, std::vector<double> correctValues)
{
	double meanError = 0.0;

	for (size_t i = 0; i < predictedValues.size(); i++) {
		meanError  +=  _lostFunction->f(predictedValues[i],correctValues[i]);
	}

	_error += meanError;
}



Layer& MLP::operator[](size_t layerIndex)
{
	return _layers[layerIndex];
}

Layer& MLP::LastLayer()
{
	size_t lastLayerIndex = _layers.size() - 1;
	return _layers[lastLayerIndex];
}



Json MLP::ToJson() const
{
	Json mlpJson;

	for (auto& layer : _layers) {
		mlpJson["MLP"].push_back(layer.ToJson());
	}

	return mlpJson;
	
}


void MLP::BuildJson()
{
	if (_outFile != "") {
		Json json = ToJson();

		std::ofstream arquivoSaida(_outFile);


		if (arquivoSaida.is_open()) {
			arquivoSaida << json.dump(4);
			arquivoSaida.close();
		} else {
			std::cerr << "\n\n[ERROR]: could not open file !!! \n\n";
		}
	}
}


