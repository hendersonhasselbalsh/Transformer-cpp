#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../../utils/basic-includes.h"


struct MatMultBackwardResult {
	Eigen::MatrixXd dL_dA;
	Eigen::MatrixXd dL_dB;

	MatMultBackwardResult() { }
};




class MatrixMultiplication {

	private:
		Eigen::MatrixXd _A;
		Eigen::MatrixXd _B;

	public:
		MatrixMultiplication();
		~MatrixMultiplication();

		Eigen::MatrixXd Forward(Eigen::MatrixXd& A, Eigen::MatrixXd& B);
		MatMultBackwardResult Backward(Eigen::MatrixXd& dL_dAB);

};



