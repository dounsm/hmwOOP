#ifndef __HMW_TYPES____
#define __HMW_TYPES____

#include <vector>
#include <functional>
#include <memory>
#include <cmath>

typedef double number;

namespace NN
{
    enum class SourceNodeType
    {
        Neuron,
        NumSource,
        None
    };

    struct SourceNode
    {
        SourceNodeType type = SourceNodeType::None;
    };

    class Synapse
    {
    public:
        Neuron *from = nullptr, *to = nullptr;
        number weight;

        number getResult(void);
    };

    class BasicNode
    {
    public:
        virtual number forward(void);
        virtual ~BasicNode();
    };

    class NumSource : public BasicNode
    {
    private:
        number num;

    public:
        number forward(void);
    };

    class Neuron : public BasicNode
    {
    private:
        std::vector<Synapse *> dendrites;
        std::unique_ptr<ActivationFunction> activationFunction;
        number bias;
    public:
        Neuron(number bias);
        number forward(void);
    };

    class Layer
    {
    private:
        std::vector<Neuron *> neurons;

    public:
        Layer(int count = 1);
    };

    class Network
    {
    private:
        std::vector<Layer *> layers;

    public:
        Network(int count = 1);
        std::shared_ptr<std::vector<number>> forward(std::vector<number> &dat);
    };

    class ActivationFunction
    {
    public:
        virtual ~ActivationFunction() = default;
        virtual number Compute(number x) const = 0;
    };

    class Linear : public ActivationFunction
    {
    public:
        number Compute(number x) const override;
    };

    class Sigmoid : public ActivationFunction
    {
    public:
        number Compute(number x) const override;
    };

    class Tanh : public ActivationFunction
    {
    public:
        number Compute(number x) const override;
    };

    class ReLU : public ActivationFunction
    {
    public:
        number Compute(number x) const override;
    };
};

#endif