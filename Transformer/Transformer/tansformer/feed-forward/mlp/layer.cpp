#pragma once

#include "layer.h"


template const auto Layer::Get<Layer::Attribute::ACCUMULATED_OUTPUTS>() const;
template const auto Layer::Get<Layer::Attribute::ALL_NEURONS>() const;
template const auto Layer::Get<Layer::Attribute::ALL_NEURONS_GRADIENTS>() const;
template const auto Layer::Get<Layer::Attribute::INPUT_SIZE>() const;
template const auto Layer::Get<Layer::Attribute::LAYER_ERRORS>() const;
template const auto Layer::Get<Layer::Attribute::LAYER_OUTPUTS>() const;
template const auto Layer::Get<Layer::Attribute::NUMBER_OF_NEURONS>() const;
template const auto Layer::Get<Layer::Attribute::OUTPUT_SIZE>() const;
template const auto Layer::Get<Layer::Attribute::RECEIVED_INPUT>() const;

template void Layer::Set<Layer::Attribute::LEARNING_RATE, double>(double value);




Layer::Layer(size_t inputSize, size_t neuronQuantity, IActivationFunction* actFun, double neuronLerningRate, ILostFunction* _lostFunc, size_t nextLayerNeuronQnt)
	: _inputSize(inputSize), _activationFunc(actFun), _learningRate(neuronLerningRate), _lostFunction(_lostFunc)
{
	_weights  =  Eigen::MatrixXd::Zero(neuronQuantity, _inputSize+1);

	XavierWeightInitialization(inputSize, neuronQuantity);

	size_t layerOutputSize = neuronQuantity + 1;
	//_outputs  =  Eigen::MatrixXd::Ones(layerOutputSize, 1);
}



Layer::~Layer() { }




std::vector<double> Layer::CalculateLayerOutputs(std::vector<double> inputs)
{
	_receivedInput  =  Utils::ReshapeMatrix( inputs, _weights.cols(), 1 );

	_weightedSums  =  WeightedSum( _weights, _receivedInput );
	Eigen::MatrixXd outputs  =  Activation( _weightedSums );

	std::vector<double> layerOutput  =  Utils::FlatMatrix( outputs );
	layerOutput.insert(layerOutput.begin(), 1.0);

	return layerOutput;
}

Eigen::MatrixXd Layer::CalculateLayerOutputs(Eigen::MatrixXd& inputs)
{
	assert(inputs.cols() == 1  &&  "[ERROR]: input must be a vertical vector!!!");

	_receivedInput  =  inputs;

	_weightedSums  =  WeightedSum(_weights, _receivedInput);
	Eigen::MatrixXd outputs  =  Activation(_weightedSums);

	std::vector<double> flatOutput  =  Utils::FlatMatrix(outputs);
	flatOutput.insert(flatOutput.begin(), 1.0);

	Eigen::MatrixXd layerOutput  =  Utils::ReshapeMatrix(flatOutput, flatOutput.size(), 1);
	return layerOutput;
}


Eigen::MatrixXd Layer::UpdateLastLayerWeight(std::vector<double> predictedValues, std::vector<double> correctValues)
{
	Eigen::MatrixXd dLoss_dActivation = LossPartialWithRespectToActivation( predictedValues,  correctValues);
	Eigen::MatrixXd dLoss_dWeightedSum = LossPartialWithRespectToWeightedSum( dLoss_dActivation);
	Eigen::MatrixXd dLoss_dWeight  =  LossPartialWithRespectToWeight( dLoss_dWeightedSum );

	_weights  =  _weights  -  _learningRate * dLoss_dWeight;

	Eigen::MatrixXd dLoss_dInput  =  LossPartialWithRespectToInput( dLoss_dWeightedSum );
	return dLoss_dInput;
}


Eigen::MatrixXd Layer::UpdateHiddenLayerWeight(Eigen::MatrixXd dLoss_dActivation)
{
	Eigen::MatrixXd dLoss_dWeightedSum = LossPartialWithRespectToWeightedSum(dLoss_dActivation);
	Eigen::MatrixXd dLoss_dWeight  =  LossPartialWithRespectToWeight(dLoss_dWeightedSum);

	_weights  =  _weights  -  _learningRate * dLoss_dWeight;

	Eigen::MatrixXd dLoss_dInput  =  LossPartialWithRespectToInput(dLoss_dWeightedSum);
	return dLoss_dInput;
}





Eigen::MatrixXd Layer::WeightedSum(Eigen::MatrixXd& weights, Eigen::MatrixXd& input)
{
	assert(weights.cols() == input.rows()  &&  "Z_[n][1] = W_[n][i] * X_[i][1] type of operation");

	Eigen::MatrixXd weightedSums  =  weights * input;

	return weightedSums;
}


Eigen::MatrixXd Layer::Activation(Eigen::MatrixXd& weightSums)
{
	assert(weightSums.cols() == 1  &&  "weightSums must be a Z_[n][1] type matrix");

	Eigen::MatrixXd activatedValues  =  Eigen::MatrixXd(weightSums.rows(), 1);

	for (size_t i = 0; i < weightSums.rows(); i++) {
		activatedValues(i,0)  =  _activationFunc->f( weightSums(i, 0) );
	}

	return activatedValues;
}



Eigen::MatrixXd Layer::LossPartialWithRespectToActivation(std::vector<double> predictedValues, std::vector<double> correctValues)
{
	Eigen::MatrixXd dLoss_dActivation  =  Eigen::MatrixXd(predictedValues.size(), 1);

	for (size_t i = 0; i < predictedValues.size(); i++) {
		dLoss_dActivation(i,0)  =  _lostFunction->df( predictedValues[i], correctValues[i] );
	}

	return dLoss_dActivation;
}


Eigen::MatrixXd Layer::LossPartialWithRespectToWeightedSum(Eigen::MatrixXd& dLoss_dActivation)
{
	assert(dLoss_dActivation.cols() == 1  &&  "dLoss_dActivation must be a M_[n][1] type matrix");
	assert(dLoss_dActivation.rows() == _weightedSums.rows()  &&  "same size for point wise operation");


	Eigen::MatrixXd dActivation_dWeightedSun  =  Eigen::MatrixXd(_weightedSums.rows(), 1);

	for (size_t i = 0; i < _weightedSums.rows(); i++) {
		dActivation_dWeightedSun(i, 0)  =  _activationFunc->df( _weightedSums(i,0) );
	}

	Eigen::MatrixXd dLoss_dWeightedSun = dLoss_dActivation.array() * dActivation_dWeightedSun.array(); // pointwise mult
	return dLoss_dWeightedSun;
}


Eigen::MatrixXd Layer::LossPartialWithRespectToWeight(Eigen::MatrixXd& dLoss_dWeightedSun)
{
	Eigen::MatrixXd dLoss_dWeight  =  dLoss_dWeightedSun  *  _receivedInput.transpose();
	return dLoss_dWeight;
}


Eigen::MatrixXd Layer::LossPartialWithRespectToInput(Eigen::MatrixXd& dLoss_dWeightedSun)
{
	size_t rows = _weights.rows();
	size_t cols = _weights.cols();

	Eigen::MatrixXd traspose_dLoss_dWeightedSun = dLoss_dWeightedSun.transpose();
	Eigen::MatrixXd weightsWithNoBias  =  _weights.block(0, 1, rows, cols-1);
	
	Eigen::MatrixXd dLoss_dInput  =  traspose_dLoss_dWeightedSun  *  weightsWithNoBias;
	return dLoss_dInput.transpose();    
}


// DEBUG THIS
void Layer::XavierWeightInitialization(size_t inputSize, size_t outputSize)
{
	double n_i = (double)inputSize;
	double n_o = (double)outputSize;

	std::random_device rd;
	std::mt19937 gen(rd());

	double range = std::sqrt(2.0 / (n_i + n_o));

	for (size_t neuronIndex = 0; neuronIndex < _weights.rows(); neuronIndex++) {
		for (size_t weightIndex = 1; weightIndex < _weights.cols(); weightIndex++) {
			//double weight  =  Utils::RandomUniformDistribution(-1.0, 1.0) * range;                    // XAVIER
			double weight  =  Utils::RandomNormalDistributionValue(-1.0, 1.0) * range;                 // XAVIER 2
			_weights(neuronIndex, weightIndex) = weight;
		}
	}
}


const double Layer::operator()(size_t neuronIndex, size_t weightIndex)
{
	return (double) _weights(neuronIndex,weightIndex);
}





Json Layer::ToJson() const
{
	size_t rows  =  _weights.rows();
	size_t cols  =  _weights.cols();

	std::vector<std::vector<double>> weightMatrix(rows, std::vector<double>(cols));

	for (int i = 0; i < _weights.rows(); ++i) {
		for (int j = 0; j < _weights.cols(); ++j) {
			weightMatrix[i][j] = _weights(i, j);
		}
	}

	Json layerJson;
	layerJson["learning-rate"] = _learningRate;
	layerJson["activation-function"] = _activationFunc->ToString();
	layerJson["neuronsQnt"] = rows;
	layerJson["weightQnt"] = cols;
	layerJson["weights"] = weightMatrix;


	return { layerJson };
}



Layer Layer::LoadWeightsFromJson(const Json& j)
{
	_activationFunc  =  Utils::StringToActivationFunction(j.at("activation-function").get<std::string>());
	_learningRate  =  j.at("learning-rate").get<double>();

	size_t neuronQnt = j.at("neuronsQnt").get<size_t>();
	size_t weightQnt = j.at("weightQnt").get<size_t>();
	const auto& weightJson = j.at("weights");

	_weights  =  Eigen::MatrixXd::Ones(neuronQnt, weightQnt);

	for (size_t i = 0; i < neuronQnt; i++) {
		for (size_t j = 0; j < weightQnt; j++) {
			_weights(i,j)  =  weightJson[i][j];
		}
	}

	return (*this);
}


