#pragma once

#include "../../../utils/basic-includes.h"



class IActivationFunction {
	public:
	virtual double f(double x) = 0;
	virtual double df(double x) = 0;
	virtual const char* ToString() = 0;
};





class CustonActivationFunction : public IActivationFunction {
	private:
	std::function<double(double x)> _function;
	double _h;

	public:
	CustonActivationFunction(std::function<double(double x)> f, double h =  1.e-11);
	~CustonActivationFunction();

	virtual double f(double x) override;
	virtual double df(double x) override;
	virtual const char* ToString() override;
};




class Sigmoid : public IActivationFunction {
	private:
		double _a;
	public:
		Sigmoid(double a = 1.00);
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};



class AdaptedSigmoid : public IActivationFunction {
	public:
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};




class Tanh : public IActivationFunction {
	public:
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};



class NormalizedTanh : public IActivationFunction {
	public:
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};



class Linear : public IActivationFunction {
	public:
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};



class ClipedLinear : public IActivationFunction {
	private:
		double _min;
		double _max;
	public:
		ClipedLinear(double min, double max);

		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};




class ReLU : public IActivationFunction {
	public:
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};




class ClipedReLU : public IActivationFunction {
	public:
		double _limit;
		ClipedReLU(double limit);
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};




class LeakyReLU : public IActivationFunction {
	private:
		
	public:
		LeakyReLU();
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};




class ParametricReLU : public IActivationFunction {
	private:
		double _param;
	public:
		ParametricReLU(double param);
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};




class GeLU : public IActivationFunction {
	public:
		GeLU();
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;

		static double phi(double x);  //  accumulated distribution
		static double pdf(double x);  //  probability density
};




class SiLU : public IActivationFunction {
	public:
		SiLU();
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;

		static double Sigmoid(double x);
		static double dSigmoid(double x);
};





class Softplus : public IActivationFunction {
	public:
		Softplus();
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};






class ELU : public IActivationFunction {
	private:
		double _a; 
	public:
		ELU(double a);
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};






class Mish : public IActivationFunction {
	public:
		Mish();
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;

		static double Softplus(double x);
		static double dSoftplus(double x);
};

