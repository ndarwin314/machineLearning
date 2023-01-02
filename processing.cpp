//
// Created by Noah Feinberg on 9/22/2022.
//

#include "processing.h"


VectorXd convert_image(const vector<unsigned char>& v) {
    VectorXd toReturn(784);
    for (int i = 0; i < 784; i++) {
        toReturn(i) = (float)v[i] / 256;
    }
    return toReturn;
}

vector<VectorXd> convert_images(const vector<vector<unsigned char>>& input) {
    vector<VectorXd> toReturn;
    toReturn.reserve(input.size());
    for (const vector<unsigned char>& v : input) {
        toReturn.emplace_back(convert_image(v));
    }
    return toReturn;
}

VectorXd convert_digit(unsigned char x) {
    VectorXd toReturn(10);
    toReturn.setZero();
    toReturn(x) = 1.0;
    return toReturn;
}

vector<VectorXd> convert_digits(const vector<unsigned char>& input) {
    vector<VectorXd> toReturn;
    toReturn.reserve(input.size());
    for (unsigned char x : input) {
        toReturn.emplace_back(convert_digit(x));
    }
    return toReturn;
}

void set_data(neural_network& network, string file) {
    auto dataset = mnist::read_dataset<>(file);
    network.set_training(convert_images(dataset.training_images), convert_digits(dataset.training_labels));
    network.set_testing(convert_images(dataset.test_images), convert_digits(dataset.test_labels));
}