#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../../utils/basic-includes.h"

#include "../feed-forward/FeedForward.h"
#include "../Scalled-Dot-Product-Attention/Scaled-Dot-Product-Attention.h"



struct AttentionHeadBackwardResult {
	Eigen::MatrixXd dL_dQ;
	Eigen::MatrixXd dL_dK;
	Eigen::MatrixXd dL_dV;

	AttentionHeadBackwardResult() { }
};

struct DL_DAttention {
	Eigen::MatrixXd Q;
	Eigen::MatrixXd K;
	Eigen::MatrixXd V;

	DL_DAttention() { }
};


class AttentionHead {

	private:
		FeedForward _linearQ;
		FeedForward _linearK;
		FeedForward _linearV;

		ScaledDotProductAttention _scaledDotProductAttention;

	public:
		AttentionHead(size_t linearSize, SDPAttention::Attrib attrib = SDPAttention::Attrib::DONT_USE_MASK, double learningRate = 0.001);
		~AttentionHead();


		Eigen::MatrixXd Forward(Eigen::MatrixXd& Q, Eigen::MatrixXd& K, Eigen::MatrixXd& V);
		//AttentionHeadBackwardResult Backward(Eigen::MatrixXd& dL_dAttentionHead);
		DL_DAttention Backward(Eigen::MatrixXd& dL_dAttentionHead);


};










