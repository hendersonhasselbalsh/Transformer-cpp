#pragma once

#include "../../utils/basic-includes.h"
#include "mlp/multy-layer-perceptron.h"



class FeedForward {

	public:
		MLP _mlp;
		Eigen::MatrixXd _receivedInput;


	public:

		FeedForward();
		~FeedForward();

		Eigen::MatrixXd Forward(Eigen::MatrixXd& inputMatrix);
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dActivationMatrix);


};
