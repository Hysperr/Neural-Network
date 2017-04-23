
#include <iostream>
#include <ctime>
#include <fstream>
#include <sstream>
#include <cassert>
#include <iomanip>
#include "Green.h"

int main() {
    srand(time(NULL));
    constexpr int num_input_nodes = 64;
    constexpr int num_output_nodes = 10;
    constexpr double learning_rate = .125;

    // hidden layer map. key layer number, value num nodes per layer. alternative can use hidden_map[k] = v notation
    std::map<int, int> hidden_map;
    hidden_map.insert(std::pair<int, int>(0, 70));

    // create neural network
    Green red(num_input_nodes, num_output_nodes, learning_rate, hidden_map, false);
    std::cout << "Bias active? " << ((red.biasIsActive()) ? "true" : "false") << '\n';

    // output identity
    std::map<int, int> output_map;
    for (int i = 0; i < num_output_nodes; i++)
        output_map.insert(std::pair<int, int>(i, i));
    red.set_output_identity(output_map);

    // training
    double label_1;
    std::vector<double> data_vector; std::string line; double i;
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

    red.clear_network();
    data_vector.clear();
    training_file.close();
    std::cout << "WEIGHTS TRAINED\n";
    int total_lines = 1797; int correct = 0; std::string line_test; double label_2, u;

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
        if (red.choose_answer(label_2))
            correct++;

        red.clear_network();
        data_vector.clear();
    }

    std::cout << "Correct: " << correct << " out of " << total_lines  << '\n';
    // Accuracy 3 different ways
    // printf
    std::printf("Accuracy: %.2f%%\n", ((double) correct / total_lines) * 100);
    // iomanip
    std::cout << "Accuracy: " << std::setprecision(2) << std::fixed << ((double) correct / total_lines) * 100 << "%\n";
    // standard print
    std::cout << "Accuracy: " << ((double) correct / total_lines) * 100  << '%' << '\n';

    return 0;
}
