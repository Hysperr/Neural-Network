
#include <iostream>
#include <ctime>
#include <fstream>
#include <sstream>
#include <cassert>
#include <iomanip>
#include <algorithm>
#include "NeuralNet.h"

int main() {
    srand(time(NULL));
    constexpr int num_input_nodes = 64;
    constexpr int num_output_nodes = 10;
    constexpr double learning_rate = .125;

    std::map<unsigned, unsigned> hidden_map;            // hidden layer map. key = layer num, value = nodes per layer
    hidden_map.insert(std::pair<int, int>(0, 150));     //  Alternative can use hidden_map[k] = v notation

    // create neural network
    NeuralNet red(num_input_nodes, num_output_nodes, learning_rate, hidden_map, false);

    // informative print
    std::cout << "Input node count = " << num_input_nodes
              << " | Output node count = " << num_output_nodes
              << " | Learning rate = " << learning_rate
              << " | Bias active? " << ((red.biasIsActive()) ? "true" : "false")
              << " \n\n";

    // output identity
    std::map<unsigned, double> output_map;
    for (int i = 0; i < num_output_nodes; i++)
        output_map.insert(std::pair<int, double>(i, i));
    red.set_output_identity(output_map);

    // training
    double label_1;
    std::vector<double> data_vector;
    std::string line;
    double i;
    std::ifstream training_file("../optdigits_train.txt");
    while (std::getline(training_file, line)) {
        std::istringstream iss(line);
        while (iss >> i) {
            data_vector.push_back(i);
            if (iss.peek() == ',') iss.ignore();
        }
        label_1 = data_vector.back();
        data_vector.pop_back();
        red.insert_data(data_vector);
        red.forward_propagate();
        red.back_propagate(label_1);
        red.clear_network();
        data_vector.clear();
    }


    training_file.close();
    std::cout << "WEIGHTS TRAINED\n";
    int total_lines = 1797;
    int correct = 0;
    std::string line_test;
    double u, label_2;


    // testing
    std::ifstream test_file("../optdigits_test.txt");
    assert(test_file.is_open());
    while (std::getline(test_file, line_test)) {
        std::istringstream iss2(line_test);
        while (iss2 >> u) {
            data_vector.push_back(u);
            if (iss2.peek() == ',') iss2.ignore();
        }
        label_2 = data_vector.back();
        data_vector.pop_back();
        red.insert_data(data_vector);
        red.forward_propagate();
        if (red.choose_answer(label_2, false))     /**< Uses default param = false. true to print network's beliefs */
            correct++;

        red.clear_network();
        data_vector.clear();
    }

    double accuracy = ((double) correct / total_lines) * 100;

    std::cout << "Correct: " << correct << " out of " << total_lines << '\n';

    std::cout << "Accuracy: " << std::setprecision(2) << std::fixed << accuracy << "%\n";

    std::cout << "Total neural network connections: " << red.get_total_connections() << '\n';


    return 0;
}
