#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../../utils/basic-includes.h"

#include "Encoder-Embeding.h"
#include "../feed-forward/FeedForward.h"
#include "../add-norm/Add-Norm.h"
#include "../Multy-Head-Attention/Attention-Head.h"
#include "../Multy-Head-Attention/Multy-Head-Attention.h"



class Encoder {

	private:
		//Embedding  _encoderEmbedding;
		
		AddNorm _attention_AddNorm;
		AddNorm _feedForward_AddNorm;

		MultyHeadAttention _multyHeadSelfAttention;

		FeedForward _feedForward;


	public:
		Encoder(size_t inputMatrixCols, size_t h, SDPAttention::Attrib attrib = SDPAttention::Attrib::DONT_USE_MASK, double learningRate = 0.001);
		~Encoder();

		Eigen::MatrixXd Forward(Eigen::MatrixXd& positionedInput);
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dAttentionStates);


};




