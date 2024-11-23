#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../../utils/basic-includes.h"

#include "../feed-forward/FeedForward.h"
#include "../add-norm/Add-Norm.h"



class Embedding {

	private:
		FeedForward _linearEmbedding;

	public:
		Embedding();
		Embedding(size_t embededWordSize, size_t dictionarySize, double learningRate = 0.001);
		~Embedding();

		Eigen::MatrixXd Forward(Eigen::MatrixXd& inputWords);
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dEmbeddedWord);

		static Eigen::MatrixXd Positioning(Eigen::MatrixXd& embededInput);

};











