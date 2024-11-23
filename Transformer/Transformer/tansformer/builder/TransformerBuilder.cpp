#include "TransformerBuilder.h"
#pragma once

TransformerBuilder::TransformerBuilder()
{
	//_transformer = EncodeDecodeTransformer(1,1,1,1, new MSE());
}

TransformerBuilder TransformerBuilder::EmbeddingSize(size_t size)
{
	_embeddingSize = size;
	return (*this);
}

TransformerBuilder TransformerBuilder::InputDictionarySize(size_t size)
{
	_inputDictionarySize = size;
	return (*this);
}

TransformerBuilder TransformerBuilder::OutputDictionarySize(size_t size)
{
	_outputDictionarySize = size;
	return (*this);
}

TransformerBuilder TransformerBuilder::Heads(size_t size)
{
	_heads = size;
	return (*this);
}

TransformerBuilder TransformerBuilder::LearningRate(double rate)
{
	_learningRate = rate;
	return (*this);
}


EncodeDecodeTransformer TransformerBuilder::Build()
{
	EncodeDecodeTransformer _transformer = EncodeDecodeTransformer(_embeddingSize, _inputDictionarySize, _outputDictionarySize, _heads, new MSE(), _learningRate);
	return _transformer;
}
