#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../../utils/basic-includes.h"



class Softmax {

	private:
		Eigen::MatrixXd _receivedInput;

	public:
		Softmax();
		~Softmax();

		Eigen::MatrixXd Forward(Eigen::MatrixXd& input);
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dSoftmax);


		static Eigen::MatrixXd MatrixSoftmax(Eigen::MatrixXd& input);

};


