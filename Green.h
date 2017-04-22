
#ifndef NEURAL_NETWORK_GREEN_NEURAL_H
#define NEURAL_NETWORK_GREEN_NEURAL_H


#include "Node.h"
#include <map>
#include <cmath>


class Green {
    /// public members
public:
    /// constructors
    Green(int num_input_nodes, int num_output_nodes, double learning_rate, std::map<int, int> &mp, bool include_bias = false);
    /// getters
    int getNum_input_nodes() const { return num_input_nodes; }
    int getNum_output_nodes() const { return num_output_nodes; }
    double getLearning_rate() const { return learning_rate; }
    std::vector<std::vector<Node>>& getNeural_obj() { return mv; }
    /// actions
    void set_output_identity(const std::map<int, int>& identity_map);
    void insert_data(const std::vector<double> &data_vector);
    void forward_propagate();
    void back_propagate(const double &label);
    void clear_network();
    bool print_best_guess(const double &label);
    /// prints
    void print_neural_layer(int index);
    void print_output_layer();
    void print_ENTIRE_network();

private:
/// private members
    bool bias;
    int num_input_nodes;
    int num_output_nodes;
    double learning_rate;
//    Node::size_type total_attachments_made;
    std::vector<std::vector<Node>> mv;
    /// private member functions (we use these to build the neural object)
    std::vector<std::vector<Node>> prepare_hidden_layers(std::map<int, int>& mp);
    void prepare_input_layer(std::vector<std::vector<Node>> &mv);
    void prepare_output_layer(std::vector<std::vector<Node>> &mv);
    void generate_neural_web(std::vector<std::vector<Node>> &mv);
    void generate_bias_nodes();
    /// their public counterparts decide which to call NB = no bias nodes, BIAS = bias nodes
    void insert_data_BIAS(const std::vector<double> &data_vector);
    void insert_data_NB(const std::vector<double> &data_vector);
    void forward_propagate_BIAS();
    void forward_propagate_NB();
    void clear_network_BIAS();
    void clear_network_NB();
    /// private member function inlined in class
    double derivative_of_sigmoid(double x){
        double num = (pow(M_E, x)) / pow((pow(M_E, x) + 1), 2);
        return num;
    }

};


#endif //NEURAL_NETWORK_GREEN_NEURAL_H
