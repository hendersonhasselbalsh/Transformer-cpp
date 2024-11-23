#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../../utils/basic-includes.h"

#include "../Encoder/Encoder-Embeding.h"
#include "../feed-forward/FeedForward.h"
#include "../add-norm/Add-Norm.h"
#include "../Multy-Head-Attention/Attention-Head.h"
#include "../Multy-Head-Attention/Multy-Head-Attention.h"




struct DL_DDecoder {
	Eigen::MatrixXd shiftedOutput;
	Eigen::MatrixXd encoderStates;
};



class Decoder {

	private:
		MultyHeadAttention _maskMultiHeadAttention;
		MultyHeadAttention _multiHeadCrossAttention;

		AddNorm _addNorm_1;
		AddNorm _addNorm_2;
		AddNorm _addNorm_3;

		FeedForward _feedForward;


	public:
		Decoder(size_t inputMatrixCols, size_t h, double learningRate = 0.001);
		~Decoder();

		Eigen::MatrixXd Forward(Eigen::MatrixXd& shiftedOutput, Eigen::MatrixXd& encoderStates);
		DL_DDecoder Backward(Eigen::MatrixXd& dL_dNormalizedOutput);


};








