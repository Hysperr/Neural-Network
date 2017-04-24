#include <iostream>
#include <ctime>
#include <fstream>
#include <sstream>
#include <cassert>
#include <iomanip>
#include <algorithm>
#include "NeuralNet.h"


int main() {
   //------------------------------------------------------------------------------------------------------------------
   // Modify net characteristics below - number of nets to create, the starting learning alpha, learning rate increment
   //------------------------------------------------------------------------------------------------------------------
   int networks_to_create = 5, ntc = networks_to_create;
   double test_learning_rate = .125, tlr = test_learning_rate;
   double lr_increment = .005;
   std::vector<double> holding_vector;
   int numRuns = 0;
   //------------------------------------------------------------------------------------------------------------------
   while (ntc > 0) {
       constexpr int num_input_nodes = 64;
       constexpr int num_output_nodes = 10;
       double learning_rate = tlr;

       // hidden layer map. key = layer number from 0, value = nodes per layer.
       // Alternative can use hidden_map[k] = v notation
       std::map<int, int> hidden_map;
       hidden_map.insert(std::pair<int, int>(0, 70));      /**< Add more layers if desired **/

       // create neural network
       NeuralNet red(num_input_nodes, num_output_nodes, learning_rate, hidden_map, false);     /**< Default param false. true for bias node option **/

       // informative print
       std::cout << "Input node count = " << num_input_nodes
                 << " | Output node count = " << num_output_nodes
                 << " | Learning rate = " << learning_rate
                 << " | Bias active? " << ((red.biasIsActive()) ? "true" : "false")
                 << " \n";

       // output identity
       std::map<int, double> output_map;
       for (int i = 0; i < num_output_nodes; i++)
           output_map.insert(std::pair<int, double>(i, i));
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
           if (red.choose_answer(label_2, false))     /**< Uses default param = false. true to print network's beliefs **/
               correct++;

           red.clear_network();
           data_vector.clear();
       }


       double accuracy = ((double) correct / total_lines) * 100;
       std::cout << "Correct: " << correct << " out of " << total_lines  << '\n';
       std::printf("Accuracy: %.2f%%\n", accuracy);
       std::cout << "Run #" << numRuns++ << std::endl;
       std::cout << '\n';
       holding_vector.push_back(accuracy);
       tlr += lr_increment;
       ntc--;
   }



   auto macc_it = std::max_element(holding_vector.begin(), holding_vector.end());
   int run_of_macc = (int) (macc_it - holding_vector.begin());
   double lr_of_macc = test_learning_rate + (lr_increment * run_of_macc);
   std::cout << "Max acc: " << *macc_it << std::endl;
   std::cout << "Learning rate when max found: " << lr_of_macc << std::endl;
   std::cout << "Check run: " << run_of_macc << std::endl;




   return 0;
}

