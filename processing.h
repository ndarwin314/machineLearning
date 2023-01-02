//
// Created by Noah Feinberg on 9/22/2022.
//

#ifndef MACHINELEARNING_PROCESSING_H
#define MACHINELEARNING_PROCESSING_H
#include <vector>
#include <Eigen/Dense>
#include "mnist/mnist_reader.hpp"
#include "neural_network.h"
using namespace std;
using namespace Eigen;

VectorXd convert_image(const vector<unsigned char>&);
vector<VectorXd> convert_images(const vector<vector<unsigned char>>&);
VectorXd convert_digit(unsigned char x);
vector<VectorXd> convert_digits(const vector<unsigned char>& input);
void set_data(neural_network&, string);


#endif //MACHINELEARNING_PROCESSING_H
