#pragma once


//#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "basic-includes.h"
#include "../tansformer/feed-forward/mlp/activation-functions.h"

#define INPUT first
#define LABEL second

using MLPTrainigData = std::pair<std::vector<double>, std::vector<double>>;
using CNNTrainigData = std::pair<Eigen::MatrixXd, std::vector<double>>;

class MLP_DATA_CLASS {
	public:
		virtual std::vector<double> Label() = 0;
		virtual std::vector<double> Input() = 0;
};

struct MLP_DATA {
	std::vector<double> input;
	size_t labelIndex;

	MLP_DATA() {}
	MLP_DATA(std::vector<double> i, size_t l) : input(i), labelIndex(l) { }
};


struct CNN_DATA {
	Eigen::MatrixXd input;
	std::vector<double> label;
	size_t labelIndex;

	CNN_DATA() { }
	CNN_DATA(Eigen::MatrixXd i, size_t l) : input(i), labelIndex(l) { }
};


namespace Utils {

	double RandomNormalDistributionValue(double min, double max);

	double RandomUniformDistribution(double min, double max);
	double ScalarProduct(std::vector<double> inputs, std::vector<double> weights);
	double Normalize(double value, double min, double max, double initial = -1.0, double final = 1.0);
	std::vector<double> BatchNormalization(std::vector<double> originalInput);
	double Mean(std::vector<double> values);
	
	double Variance(std::vector<double> values, double mean);


	std::vector<double> FlatMatrix(Eigen::MatrixXd input);
	Eigen::MatrixXd ReshapeMatrix(std::vector<double> gradients, size_t rows, size_t cols);

	std::vector<std::string> SplitString(const std::string& input, const std::string& delimiter);

	
	IActivationFunction* StringToActivationFunction(std::string functionName);

	std::vector<double> Add(std::vector<double> a, std::vector<double> b);


	std::vector<std::vector<MLPTrainigData>> ShuffleBatch(std::vector<std::vector<MLPTrainigData>> batch);
	std::vector<std::vector<MLPTrainigData>> ShuffleBatch(std::vector<MLP_DATA> trainigSet, size_t batchSize, std::function<std::vector<double>(size_t)> ParseLabelToVector);
	std::vector<std::vector<MLPTrainigData>> ShuffleBatch(std::vector<std::vector<MLPTrainigData>> batchs, size_t batchSize);



	void CalculateMeanVector(std::vector<MLPTrainigData> trainigSet, std::vector<double>* meansResult);
	void CalculateDeviationVector(std::vector<MLPTrainigData> trainigSet, std::vector<double>* means, std::vector<double>* deviationsResult);

	void CalculateMeanVector(std::vector<MLP_DATA> trainigSet, std::vector<double>* meansResult);
	void CalculateDeviationVector(std::vector<MLP_DATA> trainigSet, std::vector<double>* means, std::vector<double>* deviationsResult);

	void BatchNorm(std::vector<double>* inputs, std::vector<double>* means, std::vector<double>* devs, double* alpha = nullptr, double* beta = nullptr);

	void DataNorm(std::vector<double>* inputs, std::vector<double>* means, std::vector<double>* devs);


	void ScalateAndShift(std::vector<double>* inputs, double* alpha = nullptr, double* beta = nullptr);

}

namespace Utils {

	Eigen::MatrixXd Rotate_180Degree(Eigen::MatrixXd& matrix);

}

namespace Utils{
	Eigen::MatrixXd ScalateMatrix(Eigen::MatrixXd& mat, double a, double b);
}


