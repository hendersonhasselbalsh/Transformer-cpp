#include "Softmax.h"

Softmax::Softmax()
{
}

Softmax::~Softmax()
{
}



Eigen::MatrixXd Softmax::Forward(Eigen::MatrixXd& input)
{
	_receivedInput = input;
    Eigen::MatrixXd softmaxResult  =  MatrixSoftmax( _receivedInput );
    return softmaxResult;
}



Eigen::MatrixXd Softmax::Backward(Eigen::MatrixXd& dL_dSoftmax)
{
	assert(dL_dSoftmax.rows()==_receivedInput.rows() && dL_dSoftmax.cols()==_receivedInput.cols()  &&  "[ERROR]: both matrix must have same dimentions");

    Eigen::MatrixXd dL_dInput  =  Eigen::MatrixXd(dL_dSoftmax.rows(), dL_dSoftmax.cols());
    Eigen::MatrixXd softmaxResult  =  MatrixSoftmax(_receivedInput);

    for (int i = 0; i < _receivedInput.rows(); i++) {
        for (int j = 0; j < _receivedInput.cols(); j++) {
            double s  =  softmaxResult(i,j);

            double dSoftmax_dInputElement  =  s * (1.0 - s);
            dL_dInput(i,j)  =  dL_dSoftmax(i,j) * dSoftmax_dInputElement;
        }
    }

    return dL_dInput;
}

/*           ORIGINAL
Eigen::MatrixXd Softmax::Backward(Eigen::MatrixXd& dL_dSoftmax)
{
    assert(dL_dSoftmax.rows()==_receivedInput.rows() && dL_dSoftmax.cols()==_receivedInput.cols()  &&  "[ERROR]: both matrix must have same dimentions");

    Eigen::MatrixXd dL_dInput  =  Eigen::MatrixXd(dL_dSoftmax.rows(), dL_dSoftmax.cols());
    Eigen::MatrixXd softmaxResult  =  MatrixSoftmax(_receivedInput);

    for (int i = 0; i < _receivedInput.rows(); i++) {
        for (int j = 0; j < _receivedInput.cols(); j++) {
            double s  =  softmaxResult(i,j);

            double dSoftmax_dInputElement  =  s * (1.0 - s);
            dL_dInput(i,j)  =  dL_dSoftmax(i,j) * dSoftmax_dInputElement;
        }
    }

    return dL_dInput;
}
*/




Eigen::MatrixXd Softmax::MatrixSoftmax(Eigen::MatrixXd& input)
{
    Eigen::MatrixXd result(input.rows(), input.cols());

    for (int i = 0; i < input.rows(); i++) {
        Eigen::VectorXd rowExp = input.row(i).array().exp(); 

        double sumExp = rowExp.sum();

        result.row(i) = rowExp / sumExp;
    }

    return result;


    /*Eigen::MatrixXd result = Eigen::MatrixXd(input.rows(), input.cols());

    double sum_exp = input.unaryExpr([](double x) { return std::exp(x); }).sum();

    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            double exp_element = std::exp( input(i, j) );
            result(i, j) = exp_element / sum_exp;
        }
    }*/

    return result;
}
