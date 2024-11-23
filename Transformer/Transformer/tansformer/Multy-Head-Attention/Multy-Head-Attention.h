#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../../utils/basic-includes.h"

#include "Attention-Head.h"



//struct Head {
//	Eigen::MatrixXd Q;
//	Eigen::MatrixXd K;
//	Eigen::MatrixXd V;
//
//	Head() { }
//};

//struct DL_DHead {
//	Eigen::MatrixXd Q;
//	Eigen::MatrixXd K;
//	Eigen::MatrixXd V;
//
//	DL_DHead(){ }
//};



class MultyHeadAttention {

	private:
		std::vector<AttentionHead> _attentionHeads;
		FeedForward _linear;

		size_t _h;


	public:
		MultyHeadAttention();
		MultyHeadAttention(size_t inputMatrixCols, size_t h, SDPAttention::Attrib attrib = SDPAttention::Attrib::DONT_USE_MASK, double learningRate = 0.001);
		~MultyHeadAttention();


		Eigen::MatrixXd Forward(Eigen::MatrixXd& Q, Eigen::MatrixXd& K, Eigen::MatrixXd& V);
		DL_DAttention Backward(Eigen::MatrixXd& dL_dMultyHeadAttention);


		std::vector<DL_DAttention> SplitMatrixIntoHeads(Eigen::MatrixXd& input, size_t h);
		DL_DAttention ConcatMatrixIntoHeads(std::vector<DL_DAttention> heads);

		std::vector<Eigen::MatrixXd> SplitMatrix(Eigen::MatrixXd& input, size_t h);
		Eigen::MatrixXd ConcatMatrix(std::vector<Eigen::MatrixXd> matrixies);
		
};





