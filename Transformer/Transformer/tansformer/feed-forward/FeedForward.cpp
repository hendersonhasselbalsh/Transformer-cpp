#include "FeedForward.h"

FeedForward::FeedForward()
{
}

FeedForward::~FeedForward()
{
}

Eigen::MatrixXd FeedForward::Forward(Eigen::MatrixXd& inputMatrix)
{
	_receivedInput = inputMatrix;

	size_t outputRows  =  inputMatrix.rows();
	size_t outputCols  =  _mlp.Get<MLP::Attribute::OUTPUT_SIZE>();

	Eigen::MatrixXd output = Eigen::MatrixXd(outputRows, outputCols);


	for (size_t i = 0; i < inputMatrix.rows(); i++) {
		Eigen::MatrixXd inputRow = Eigen::MatrixXd::Ones(inputMatrix.cols()+1, 1);
		inputRow.block(1, 0, inputMatrix.cols(), 1) = inputMatrix.row(i).transpose();


		Eigen::MatrixXd outputRow = _mlp.Foward( inputRow );


		outputRow.transposeInPlace();
		output.row(i) = outputRow;
	}

	return output;
}


Eigen::MatrixXd FeedForward::Backward(Eigen::MatrixXd& dL_dActivationMatrix)
{
	assert(_receivedInput.rows() == dL_dActivationMatrix.rows() && "[ERROR]: both matrix must have same row size");


	size_t matRows  =  _receivedInput.rows();
	size_t matCols  =  _receivedInput.cols();

	Eigen::MatrixXd dL_dInputMatrix = Eigen::MatrixXd(matRows, matCols);


	for (size_t i = 0; i < _receivedInput.rows(); i++) {
		Eigen::MatrixXd inputRow = Eigen::MatrixXd::Ones(_receivedInput.cols()+1, 1);
		inputRow.block(1, 0, _receivedInput.cols(), 1) = _receivedInput.row(i).transpose();

		Eigen::MatrixXd dL_dActivation = dL_dActivationMatrix.row(i).transpose();


		Eigen::MatrixXd outputRow = _mlp.Foward( inputRow );
		Eigen::MatrixXd dL_dInput = _mlp.Backward( dL_dActivation );


		dL_dInput.transposeInPlace();
		dL_dInputMatrix.row(i) = dL_dInput;
	}

	return dL_dInputMatrix;
}
