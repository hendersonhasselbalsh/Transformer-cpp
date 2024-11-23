#pragma once

#include "../../../utils/basic-includes.h"
#include "layer.h"

#define INPUT first
#define LABEL second
using MLPTrainigData = std::pair<std::vector<double>, std::vector<double>>;


class MlpBuilder;
class FeedForward;



class MLP {

	private:
	//--- atributos importantes do mlp
		std::vector<Layer> _layers;
		ILostFunction* _lostFunction;

		std::function<std::vector<double>(size_t)> ParseLabelToVector;
		std::function<void(size_t, double, double&)> UpdateLeraningRate;                      //  double f(size_t epoch, double accuracy, double currentLearningRate)
		std::string _outFile;

		size_t _inputSize;
		size_t _maxEpochs;
		double _acceptableAccuracy;
		double _error;
		std::vector<double> _accumulatedGradients;


	//--- construtor privado (usado pelo builder)
		MLP();


	//--- backward and forward
		std::vector<double> Foward(std::vector<double> input);
		std::vector<double> Backward(std::vector<double> predictedValues, std::vector<double> correctValues);
		std::vector<double> Backward(std::vector<double> lossGradientWithRespectToOutput);

		Eigen::MatrixXd Foward(Eigen::MatrixXd& input);
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dActivation);
		
		void BuildJson();
		Json ToJson() const;

		void ChangeLearningRate(size_t epoch, double error);
		void CalculateError(std::vector<double> predictedValues, std::vector<double> correctValues);
		

	public:
		~MLP();

		void Training(std::vector<MLPTrainigData> trainigSet, std::function<void(void)> callback = [](){} );
		void Training(std::vector<MLP_DATA> trainigSet, std::function<void(void)> callback = [](){} );


		std::vector<double> Classify(std::vector<double> input);
		size_t Classify(std::vector<double> input, std::function<size_t(std::vector<double>)> ParseOutputToLabel);
		void Classify(std::vector<std::vector<double>> inputSet, std::function<void(std::vector<double>)> CallBack);
		void Classify(std::vector<MLP_DATA> inputSet, std::function<void(std::vector<double>)> CallBack);


		Layer& operator[](size_t layerIndex);
		Layer& LastLayer();

		enum class Attribute {
			LAST_LAYER,
			OUTPUT_SIZE
		};
		template <MLP::Attribute attrib> const auto Get() const;

		friend class MlpBuilder;
		friend class FeedForward;

};



template<MLP::Attribute attrib>
inline const auto MLP::Get() const
{
	if constexpr (attrib == MLP::Attribute::LAST_LAYER) {
		size_t lastLayerIndex = _layers.size() - 1;
		return _layers[lastLayerIndex];
	}
	else if constexpr (attrib == MLP::Attribute::OUTPUT_SIZE) {
		size_t lastLayerIndex = _layers.size() - 1;
		size_t numberOfNeurons = _layers[lastLayerIndex]._weights.rows();
		return numberOfNeurons;
	}
	else {
		assert(false && "cant get this attribute");
	}
}





#include "mlp-builder.h"
