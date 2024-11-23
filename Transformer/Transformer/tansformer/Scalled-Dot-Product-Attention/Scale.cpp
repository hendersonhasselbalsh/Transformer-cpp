#include "Scale.h"

Scale::Scale()
{
	_dk = 1.0;
}

Scale::Scale(double keyDimension)
	: _dk(keyDimension)
{
}

Scale::~Scale()
{
}



Eigen::MatrixXd Scale::Forward(Eigen::MatrixXd& input)
{
	Eigen::MatrixXd result  =  input * (1.0/std::sqrt(_dk));
	return result;
}




Eigen::MatrixXd Scale::Backward(Eigen::MatrixXd& dL_dScale)
{
	Eigen::MatrixXd dL_dInput  =  dL_dScale * (1.0/std::sqrt(_dk));
	return dL_dInput;
}
