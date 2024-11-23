#pragma once

#include "../../../utils/basic-includes.h"
#include "multy-layer-perceptron.h"



class DenseLayer {

	public:
		size_t _qntNeurons;
		IActivationFunction* _activationFunction;
		ILostFunction* _lostFunction;
		double _learningRate;

		DenseLayer(size_t qntNeurons, IActivationFunction* activationFunction = new Tanh(), double learningRate = 0.01, ILostFunction* lostFunction = nullptr)
			: _qntNeurons(qntNeurons), _activationFunction(activationFunction), _learningRate(learningRate), _lostFunction(lostFunction)
		{ }

};




class MlpBuilder {

	private:
		size_t inputSize;
		MLP _mlp;

	public:
		MlpBuilder();
		
		MLP Build();

		MlpBuilder InputSize( size_t size );
		MlpBuilder Architecture( std::vector<size_t> neuronsInLayer );
		MlpBuilder Architecture( std::vector<DenseLayer> layerSignature );
		MlpBuilder LostFunction( ILostFunction* lostFunction );
		MlpBuilder MaxEpochs( size_t epochs );
		MlpBuilder AcceptableAccuracy( double accuracy );
		MlpBuilder ParseLabelToVector( std::function<std::vector<double>(size_t)> CallBack );
		MlpBuilder SaveOn(std::string outFile);
		MlpBuilder LoadArchitectureFromJson(std::string file);
		//MlpBuilder WhenToUpdateLearningRate(std::function<bool(size_t, double)> Conddition);
		MlpBuilder UpdateLearningRate(std::function<double(size_t, double, double&)> func);   // epoch, error, learnrate

};


