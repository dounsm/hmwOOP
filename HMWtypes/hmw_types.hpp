#ifndef __HMW_TYPES____
#define __HMW_TYPES____

#include <vector>
#include <functional>
#include <memory>
#include <cmath>

typedef double number;

namespace NN{
    enum class SourceNodeType {
        Neuron,NumSource,None
    };
    
    struct SourceNode {
        SourceNodeType type = SourceNodeType::None;

    };
    
    class Synapse{
        public:
            Neuron *from = nullptr,*to = nullptr;
            number weight;

            number getResult(void);

    };

    class BasicNode{
        public:
            virtual number forward(void);
            virtual ~BasicNode();
    };

    class NumSource: public BasicNode{
        private:
            number num;
        public:
            number forward(void);
    };

    class Neuron: public BasicNode{
        private:
            std::vector<Synapse*> dendrites;
            std::function<number(number)> activationFunction;
        public:
            Neuron();
            number forward(void);
    };

    class Layer{
        private:
            std::vector<Neuron*> neurons;
        public:
            Layer(int count = 1);
    };

    class Network{
        private:
            std::vector<Layer*> layers;
        public:
            Network(int count = 1);
            std::shared_ptr<std::vector<number>> forward(std::vector<number>& dat);
    };

    struct Activations {
    // Sigmoid
    static inline std::function<number(number)> sigmoid =
        [](number x) -> number {
            return number(1) / (number(1) + std::exp(-x));
        };

    // Tanh
    static inline std::function<number(number)> tanh =
        [](number x) -> number {
            return std::tanh(x);
        };

    // ReLU
    static inline std::function<number(number)> relu =
        [](number x) -> number {
            return x > number(0) ? x : number(0);
        };
};
};

#endif