#include "Scaled-Dot-Product-Attention.h"

ScaledDotProductAttention::ScaledDotProductAttention(SDPAttention::Attrib attrib)
{
	if (attrib == SDPAttention::Attrib::DONT_USE_MASK) {
		_useMask = false;
	} else {
		_useMask = true;
	}

	_softmax = Softmax();
	_dotProductMatrixMultiplication = MatrixMultiplication();
	_contextVectorMatrixMultiplication = MatrixMultiplication();

	_scale = Scale(1.0); // _scale is officialy initialized during forward
}



ScaledDotProductAttention::~ScaledDotProductAttention()
{
}




Eigen::MatrixXd ScaledDotProductAttention::Forward(Eigen::MatrixXd& querry, Eigen::MatrixXd& key, Eigen::MatrixXd& value)
{
	_scale  =  Scale( (double)(key.cols()) );

	Eigen::MatrixXd transposeKey  =  key.transpose();
	Eigen::MatrixXd dotProduct  =  _dotProductMatrixMultiplication.Forward(querry, transposeKey);
	Eigen::MatrixXd scaledDotProduct  =  _scale.Forward(dotProduct);

	if (_useMask == true) {
		Eigen::MatrixXd mask = BuildMask(scaledDotProduct.rows(), scaledDotProduct.cols());
		scaledDotProduct  =  scaledDotProduct  +  mask;
	}

	Eigen::MatrixXd attentionScore  =  _softmax.Forward( scaledDotProduct );
	Eigen::MatrixXd attentionHead  =  _contextVectorMatrixMultiplication.Forward( attentionScore, value );

	return attentionHead;
}



SDPAttentionBackwardResult ScaledDotProductAttention::Backward(Eigen::MatrixXd& dL_dAttentionHead)
{
	SDPAttentionBackwardResult backwardResult  =  SDPAttentionBackwardResult();


	MatMultBackwardResult contextVectorBackwardResult = _contextVectorMatrixMultiplication.Backward(dL_dAttentionHead);
	Eigen::MatrixXd& dL_dAttentionScore  =  contextVectorBackwardResult.dL_dA;
	//backwardResult.dL_dValue  =  contextVectorBackwardResult.dL_dB;  // previous
	backwardResult.dL_dValue  =  contextVectorBackwardResult.dL_dB.transpose();

	Eigen::MatrixXd dL_dScaledDotProduct  =  _softmax.Backward( dL_dAttentionScore );
	Eigen::MatrixXd dL_dDotProduct  =  _scale.Backward( dL_dScaledDotProduct ); 
	MatMultBackwardResult dotProductBackwardResult  =  _dotProductMatrixMultiplication.Backward( dL_dDotProduct ); 

	backwardResult.dL_dQuery  =  dotProductBackwardResult.dL_dA;
	//backwardResult.dL_dKey  =  dotProductBackwardResult.dL_dB.transpose();  // previous
	backwardResult.dL_dKey  =  dotProductBackwardResult.dL_dB;

	return backwardResult;
}






Eigen::MatrixXd ScaledDotProductAttention::BuildMask(size_t rows, size_t cols)
{
	Eigen::MatrixXd mask  =  Eigen::MatrixXd(rows, cols);

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {

			if (j > i) {
				mask(i, j)  =  -std::numeric_limits<double>::infinity();
			} else {
				mask(i, j) = 0.0;
			}

		}
	}

	return mask;
}


