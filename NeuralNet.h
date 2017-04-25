
#ifndef NEURAL_NETWORK_GREEN_NEURAL_H
#define NEURAL_NETWORK_GREEN_NEURAL_H


#include "Node.h"
#include <map>
#include <cmath>

class NeuralNet {
public:
    /// ctors
    NeuralNet(unsigned int num_input_nodes, unsigned int num_output_nodes, double learning_rate,
              std::map<unsigned, unsigned> &mp,
              bool include_bias = false);
    /// getters
    const std::vector<std::vector<Node>>& getNeural_obj() const { return mv; }
    Node::size_type get_total_connections() const { return total_conn; }
    int getNum_output_nodes() const { return num_output_nodes; }
    int getNum_input_nodes() const { return num_input_nodes; }
    double getLearning_rate() const { return learning_rate; }
    bool biasIsActive() const { return bias; }
    /// operations
    void set_output_identity(const std::map<unsigned, double> &identity_map, bool debug_print = false);
    void forward_propagate() { (bias) ? forward_propagate_BIAS() : forward_propagate_NB(); }
    void clear_network() { (bias) ? clear_network_BIAS() : clear_network_NB(); }
    bool choose_answer(const double label, bool debug_print = false) const;
    void insert_data(const std::vector<double> &data_vector);
    void back_propagate(const double label);
    /// prints
    void print_neural_layer(int index) const;
    void print_ENTIRE_network() const;
    void print_output_layer() const;
    void print_input_layer() const;

private:
/// private members
    bool bias;
    unsigned int num_input_nodes;
    unsigned int num_output_nodes;
    double learning_rate;
    Node::size_type total_conn;
    std::vector<std::vector<Node>> mv;
    /// private member functions (we use these to build the neural object)
    std::vector<std::vector<Node>> prepare_hidden_layers(std::map<unsigned, unsigned> &mp);
    void prepare_input_layer();
    void prepare_output_layer();
    void generate_neural_web();
    void generate_bias_nodes();
    /// their public counterparts decide which to call NB = no bias nodes, BIAS = bias nodes
    void forward_propagate_BIAS();
    void forward_propagate_NB();
    void clear_network_BIAS();
    void clear_network_NB();
    /// private member function inlined in class
    double derivative_of_sigmoid(double x){
        return (pow(M_E, x)) / pow((pow(M_E, x) + 1), 2);
    }

};


#endif //NEURAL_NETWORK_GREEN_NEURAL_H
