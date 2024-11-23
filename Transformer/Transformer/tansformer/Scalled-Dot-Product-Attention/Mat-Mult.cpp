#include "Mat-Mult.h"

MatrixMultiplication::MatrixMultiplication()
{
}

MatrixMultiplication::~MatrixMultiplication()
{
}




Eigen::MatrixXd MatrixMultiplication::Forward(Eigen::MatrixXd& A, Eigen::MatrixXd& B)
{
	assert(A.cols() == B.rows()  &&  "[ERROR]: A[i,j] x B[j,k]  =  AB[i,k]");

	_A = A;
	_B = B;

	return (_A * _B);
}


MatMultBackwardResult MatrixMultiplication::Backward(Eigen::MatrixXd& dL_dAB)
{
	MatMultBackwardResult backwardResult = MatMultBackwardResult();

	backwardResult.dL_dA  =  dL_dAB * _B.transpose();
	backwardResult.dL_dB  =  dL_dAB.transpose() * _A;

	return backwardResult;
}
