#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../../utils/basic-includes.h"

#include "Mat-Mult.h"
#include "Softmax.h"
#include "Scale.h"

#define SDPAttention ScaledDotProductAttention


struct SDPAttentionBackwardResult {
	Eigen::MatrixXd dL_dQuery;
	Eigen::MatrixXd dL_dKey;
	Eigen::MatrixXd dL_dValue;

	SDPAttentionBackwardResult() { }
};



class ScaledDotProductAttention {

	private:
		bool _useMask;
		MatrixMultiplication _dotProductMatrixMultiplication;
		MatrixMultiplication _contextVectorMatrixMultiplication;
		Softmax _softmax;
		Scale _scale;


	public:
		enum Attrib { USE_MASK, DONT_USE_MASK };

		ScaledDotProductAttention(SDPAttention::Attrib attrib = SDPAttention::Attrib::DONT_USE_MASK);
		~ScaledDotProductAttention();

		Eigen::MatrixXd Forward(Eigen::MatrixXd& querry, Eigen::MatrixXd& key, Eigen::MatrixXd& value);
		SDPAttentionBackwardResult Backward(Eigen::MatrixXd& dL_dAttentionHead);


		Eigen::MatrixXd BuildMask(size_t rows, size_t cols); 

};








