//
// Created by Noah Feinberg on 9/19/2022.
//

#ifndef NEURALNETWOEK_NEURAL_NETWORK_H
#define NEURALNETWOEK_NEURAL_NETWORK_H
#include <vector>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;


class neural_network {
    vector<VectorXd> biases;
    vector<MatrixXd> weights;
    vector<VectorXd> training_data, training_labels, test_data, test_labels;
    int inputSize, ouputSize, l, epoch;
    float learningRate, decay;

    VectorXd activation(int, const VectorXd&);

    public:
        neural_network(int, int, vector<int>);
        VectorXd evaluate(const VectorXd&);
        void backprop(const VectorXd&, const VectorXd&);
        void backprop();
        void set_training(vector<VectorXd>, vector<VectorXd>);
        void set_testing(vector<VectorXd>, vector<VectorXd>);
        double test_loss();
};

template<typename T>
T relu(T x) {
    return std::max((T) 0, x);
}

template<typename T>
double relu_derivative(T x) {
    return x > 0 ? 1: 0;
}
#endif //NEURALNETWOEK_NEURAL_NETWORK_H
