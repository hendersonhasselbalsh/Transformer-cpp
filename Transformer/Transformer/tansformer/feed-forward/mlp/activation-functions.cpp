#pragma once

#include "activation-functions.h"


///-------------
///  custom
///-------------

CustonActivationFunction::CustonActivationFunction(std::function<double(double x)> f, double h)
    : _function(f), _h(h)
{
}

CustonActivationFunction::~CustonActivationFunction()
{
}

double CustonActivationFunction::f(double x)
{
    return _function(x);
}

double CustonActivationFunction::df(double x)
{
    // metodo das diferen�as finitas central de ordem h^2
    double df = (_function(x + _h)  -  _function(x - _h)) / (2.00 * _h);
    return df;
}

const char* CustonActivationFunction::ToString()
{
    return "custonMade";
}




///-------------
///  sigmoid
///-------------

Sigmoid::Sigmoid(double a)
{
    _a = a;
}

double Sigmoid::f(double x)
{
    return 1.0 / (1.0 + std::exp(-x*_a));
}

double Sigmoid::df(double x)
{
    double sig_x = f(x);
    return _a * sig_x * (1.0 - sig_x);
}

const char* Sigmoid::ToString()
{
    return "Sigmoid";
}




///-------------
///  Tanh
///-------------

double Tanh::f(double x)
{
    return std::tanh(x);
}

double Tanh::df(double x)
{
    double tanh_x = std::tanh(x);
    return 1.0 - tanh_x * tanh_x;
}

const char* Tanh::ToString()
{
    return "Tanh";
}



///------------------
///  NormalizedTanh
///------------------

double NormalizedTanh::f(double x)
{
    return (std::tanh(x) + 1.0) / 2.0;
}

double NormalizedTanh::df(double x)
{
    double tanh_x = tanh(x);
    return (1.0 - tanh_x * tanh_x) / 2.0;
}

const char* NormalizedTanh::ToString()
{
    return "NormalizedTanh";
}



///------------------
///  Linear
///------------------

double Linear::f(double x)
{
    return x;
}

double Linear::df(double x)
{
    return 1.0;
}

const char* Linear::ToString()
{
    return "Linear";
}



///------------------
///  Sigmoid
///------------------

double AdaptedSigmoid::f(double x)
{
    Sigmoid sigmoid;
    return (2.0 * sigmoid.f(x)) - 1.0;
}

double AdaptedSigmoid::df(double x)
{
    Sigmoid sigmoid;
    return 2.0 * std::exp(-x)  * sigmoid.f(x) * sigmoid.f(x);
}

const char* AdaptedSigmoid::ToString()
{
    return "AdaptedSigmoid";
}




///------------------
///  ClipedLinear
///------------------

ClipedLinear::ClipedLinear(double min, double max)
    : _min(min), _max(max)
{
}

double ClipedLinear::f(double x)
{
    if (x > _max) { return _max; }
    if (x < _min) { return _min; }
    return x;
}

double ClipedLinear::df(double x)
{
    if (x < _min || x > _max) { return 0; }
    return 1.0;
}

const char* ClipedLinear::ToString()
{
    return "ClipedLinear";
}



///-------------
///  ReLU
///-------------

double ReLU::f(double x)
{
    return std::max(0.0, x);
}

double ReLU::df(double x)
{
    if (x < 0) { return 0.0; } else { return 1.0; };
}

const char* ReLU::ToString()
{
    return "ReLU";
}




///-------------
///  LeakyReLU
///-------------

LeakyReLU::LeakyReLU()
{
}

double LeakyReLU::f(double x)
{
    if (x >= 0 ) { return x; }
    else { return 0.01 * x; }
}

double LeakyReLU::df(double x)
{
    if (x < 0) { return 0.01; } else { return 1.0; };
}

const char* LeakyReLU::ToString()
{
    return "LeakyReLU";
}





///-------------
///  ParametricReLU
///-------------

ParametricReLU::ParametricReLU(double param)
    : _param(param)
{
}

double ParametricReLU::f(double x)
{
    if (x >= 0) { return x; } 
    else { return _param * x; }
}

double ParametricReLU::df(double x)
{
    if (x < 0) { return _param; } else { return 1.0; };
}

const char* ParametricReLU::ToString()
{
    return "ParametricReLU";
}





///-------------
///  GeLU
///-------------

GeLU::GeLU()
{
}

double GeLU::f(double x)
{
    return x * phi(x);
}

double GeLU::df(double x)
{
    return phi(x) + x * pdf(x);
}

const char* GeLU::ToString()
{
    return nullptr;
}


double GeLU::phi(double x)
{
    return 0.5 * erfc(-x / std::sqrt(2.0));
}

double GeLU::pdf(double x)
{
    double PI = (double)M_PI;
    return ( 1.0/std::sqrt(2.0 * PI) ) * std::exp(-0.5 * x * x);
}






///-------------
///  SiLU
///-------------

SiLU::SiLU()
{
}

double SiLU::f(double x)
{
    return x * SiLU::Sigmoid(x);
}

double SiLU::df(double x)
{
    return x * SiLU::dSigmoid(x) + SiLU::Sigmoid(x);
}

const char* SiLU::ToString()
{
    return nullptr;
}

double SiLU::Sigmoid(double x)
{
    return (1.0 / (1.0 + std::exp(-x)) );
}

double SiLU::dSigmoid(double x)
{
    return SiLU::Sigmoid(x) * (1.0 - SiLU::Sigmoid(x));
}







///-------------
///  Softplus / Smooth ReLU
///-------------

Softplus::Softplus()
{
}

double Softplus::f(double x)
{
    if(x == 0.0) { return std::log(2.0); }
    return std::log(1.0 + std::exp(x));
}

double Softplus::df(double x)
{
    return (1.0 / (1.0 + std::exp(-x)));
}

const char* Softplus::ToString()
{
    return nullptr;
}




///-------------
///  Softplus / Smooth ReLU
///-------------

ELU::ELU(double a)
    : _a(a)
{
}

double ELU::f(double x)
{
    if (x > 0) { return x; }
    else { return _a * (std::exp(x - 1.0)); }
}

double ELU::df(double x)
{
    if (x > 0) { return 1.0; } 
    else { return _a * (std::exp(x)); }
}

const char* ELU::ToString()
{
    return nullptr;
}






///-------------
///  Mish
///-------------

Mish::Mish()
{
}

double Mish::f(double x)
{
    double s = Mish::Softplus(x);
    return x * std::tanh(s);
}

double Mish::df(double x)
{
    double s = Mish::Softplus(x);
    double tanh_s = std::tanh(s);
    double dsoftplus = Mish::dSoftplus(x);
    return Mish::f(x) + x * (1 - tanh_s * tanh_s) * dsoftplus;
}

const char* Mish::ToString()
{
    return nullptr;
}

double Mish::Softplus(double x)
{
    return std::log(1.0 + std::exp(x));
}

double Mish::dSoftplus(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}






///-------------
///  ClipedReLU
///-------------

ClipedReLU::ClipedReLU(double limit)
{
    _limit = limit;
}

double ClipedReLU::f(double x)
{
    if (x < 0.0) { return 0.0; } 
    else if (x > _limit) { return _limit; } 
    else { return x; }
}

double ClipedReLU::df(double x)
{
    if (x < 0.0) { return 0.0; } 
    else if (x > _limit) { return 0.0; } 
    else { return 1.0; }
}

const char* ClipedReLU::ToString()
{
    return nullptr;
}

/*

double ReLU::f(double x)
{
    return std::max(0.0, x);
}

double ReLU::df(double x)
{
    if (x < 0) { return 0.0; } else { return 1.0; };
}

const char* ReLU::ToString()
{
    return "ReLU";
}

*/
