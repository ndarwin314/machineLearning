//
// Created by Noah Feinberg on 9/19/2022.
//

#include "neural_network.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <utility>
#include "mnist/mnist_reader_less.hpp"

/*
static double sigmoid_derivative(double x) {
    double exp = std::exp(-x);
    return exp / ((1+exp)*(1+exp));
}
 */
IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
std::string sep = "\n----------------------------------------\n";

VectorXd neural_network::activation(int i, const VectorXd & input) {
    auto temp = weights[i]*input+biases[i];
    //std::cout << temp.format(CleanFmt)<<sep;
    if (i==l-1) {
        return temp.array().logistic();
    } else {
        // could be vectorized for performance
        return temp.array().unaryExpr(&relu<double>);
    }
}

neural_network::neural_network(int input, int output, vector<int> layers) {
    learningRate = 3.0;
    decay = 0.01;
    epoch = 0;
    inputSize = input;
    ouputSize = output;
    layers.insert(layers.begin(), input);
    l = layers.size();
    layers.push_back(output);
    int curr, next;
    for (int i=0; i<l; i++) {
        curr = layers[i];
        next = layers[i+1];
        VectorXd layer = VectorXd::Random(next);
        biases.push_back(layer);
        MatrixXd mat = MatrixXd::Random(next, curr);
        weights.push_back(mat);
    }
}

VectorXd neural_network::evaluate(const VectorXd& input) {
    // do size checking
    VectorXd output = input;
    for (int i=0; i<l; i++) {
        output = activation(i, output);
    }
    return output;
}

double neural_network::test_loss() {
    double loss = 0;
    int i = 0;
    for (auto v : test_data) {
        loss += (evaluate(v) - test_labels[i]).squaredNorm();
    }
    return loss / test_data.size();
}

void neural_network::backprop(const VectorXd & input, const VectorXd & target) {
    // forward output computation direction
    VectorXd current = input;
    vector<MatrixXd> outputs;
    outputs.reserve(l);
    outputs.emplace_back(current);
    for (int i=0; i<l; i++) {
        current = activation(i, current);
        outputs.emplace_back(current);
    }

    // backwards gradient computation direction
    vector<MatrixXd> weightGradients(l);
    vector<MatrixXd> biasGradients(l);
    MatrixXd weightError = ((current - target).array() *
                            target.array() * (VectorXd::Ones(target.size()) - target).array());
    weightGradients[l-1] = weightError*outputs[l-2].transpose();
    for (int i=l-1; i>0; i--) {
        weightError = outputs[i].unaryExpr(&relu_derivative<double>).array() *
                (weights[i].transpose() * weightError).array();
        weightGradients[i] = weightError * outputs[i].transpose();
    }
}
void neural_network::backprop() {
    double loss = 0;
    vector<MatrixXd> outputs(l+1);
    vector<MatrixXd> weightGradients(l);
    vector<VectorXd> biasGradients(l);
    for (int i=0; i<l; i++) {
        weightGradients[i] = MatrixXd::Zero(weights[i].rows(), weights[i].cols());
        biasGradients[i] = VectorXd::Zero(biases[i].size());
    }
    for (int j=0; j<training_data.size(); j++) {
        // forward output computation direction
        VectorXd current = training_data[j];
        VectorXd target = training_labels[j];
        outputs[0] = current;
        for (int i=0; i<l; i++) {
            current = activation(i, current);
            outputs[i+1] = current;
        }
        loss += (current - target).squaredNorm();

        // backwards gradient computation direction
        MatrixXd error = ((current - target).array() *
                          current.array() * (VectorXd::Ones(target.size()) - current).array());
        weightGradients[l-1] += error * outputs[l-1].transpose();
        biasGradients[l-1] += error;
        for (int i=l-1; i>0; i--) {
            error = outputs[i].unaryExpr(&relu_derivative<double>).array() *
                    (weights[i].transpose() * error).array();
            weightGradients[i-1] += error * outputs[i-1].transpose();
            biasGradients[i-1] += error;
        }
    }
    double r = learningRate / (1 + epoch * decay) / training_data.size();
    for (int i=0; i<weightGradients.size(); i++) {
        weights[i] -= r * weightGradients[i];
        biases[i] -= r * biasGradients[i];
    }
    std::cout<< "current loss: " << loss/60000 << endl;
}

void neural_network::set_training(vector<VectorXd> data, vector<VectorXd> labels) {
    training_data = std::move(data);
    training_labels = std::move(labels);
}

void neural_network::set_testing(vector<VectorXd> data, vector<VectorXd> labels) {
    test_data = std::move(data);
    test_labels = std::move(labels);
}
