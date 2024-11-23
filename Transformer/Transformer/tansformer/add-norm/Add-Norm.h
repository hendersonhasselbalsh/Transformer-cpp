#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../../utils/basic-includes.h"




class AddNorm {

	private:
		std::vector<double> _layerMeans;
		std::vector<double> _layerStddev;

		std::vector<double> _betas;    // layer shift
		std::vector<double> _gammas;   // layer scala

		double _learningRate;

		Eigen::MatrixXd _receivecInput;

	public:
		AddNorm();
		AddNorm(size_t embeddingSize, double learningRate = 0.001);
		~AddNorm();

		Eigen::MatrixXd Forward(Eigen::MatrixXd& inputMatrix, Eigen::MatrixXd& addMatrix);
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dNormalized);


		Eigen::MatrixXd LayerNormalization(Eigen::MatrixXd& addedMatrix);

		Eigen::MatrixXd DL_DVariance(Eigen::MatrixXd& dL_dNormalized);                               // horizontal vector
		Eigen::MatrixXd DL_DNii(Eigen::MatrixXd& dL_dVariance, Eigen::MatrixXd& dL_dNormalized);     // matrix, same dimention as input
		Eigen::MatrixXd DL_DMeans(Eigen::MatrixXd& dL_dNormalized);                                  // horizontal vector
		Eigen::MatrixXd DL_DInput(Eigen::MatrixXd& dL_dMean, Eigen::MatrixXd& dL_dNii);              // matrix, same dimention as input
		 
};




