#include <iostream>
#include "neural_network.h"
#include "mnist/mnist_reader.hpp"
#include "processing.h"
#include <chrono>
using namespace std::chrono;



int main() {
    auto start = high_resolution_clock::now();
    neural_network test = neural_network(784, 10, {10,10,10});
    set_data(test, "C:\\cpp\\mnist");
    auto input = VectorXd::Random(784);
    for (int i = 0; i < 250; i++) {
        test.backprop();
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    cout << "test loss: " << test.test_loss() << endl;
    cout << "Time taken by function: "
        << duration.count() << " seconds" << endl;
    return 0;

}
