#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../../utils/basic-includes.h"


class Scale {

	private:
		double _dk;  // Query (Q) and Key (K) vectors imension

	public:
		Scale();
		Scale(double keyDimension);
		~Scale();

		Eigen::MatrixXd Forward(Eigen::MatrixXd& input);
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dScale);

};


