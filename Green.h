
#ifndef NEURAL_NETWORK_GREEN_NEURAL_H
#define NEURAL_NETWORK_GREEN_NEURAL_H


#include "Node.h"
#include <map>
#include <cmath>


class Green {
//    friend class Node;
    /// public members
public:
    /// constructors
    Green(int num_input_nodes, int num_output_nodes, double learning_rate, std::map<int, int> &mp, bool include_bias = false);
    /// public member functions
    int getNum_input_nodes() const { return num_input_nodes; }
    int getNum_output_nodes() const { return num_output_nodes; }
    double getLearning_rate() const { return learning_rate; }
    template <class T> void insert_data(const std::vector<T> &data_vector);
    void forward_propagate();
    void back_propagate(const int label);
    void clear_network();



private:
/// private members
    bool uses_bias;
    int num_input_nodes;
    int num_output_nodes;
    double learning_rate;
    Node::size_type total_attachments_made;
    std::vector<std::vector<Node>> neural_obj;

    /// private member functions (we use these to build the neural object)
    std::vector<std::vector<Node>> prepare_hidden_layers(std::map<int, int>& mp);
    void prepare_input_layer(std::vector<std::vector<Node>> &mv);
    void prepare_output_layer(std::vector<std::vector<Node>> &mv);
    void generate_neural_web(std::vector<std::vector<Node>> &mv);
    void generate_bias_nodes();
    /// their public counterparts decide which to call NB = no bias nodes, BIAS = bias nodes
    template <class T> void insert_data_BIAS(const std::vector<T> &data_vector);
    template <class T> void insert_data_NB(const std::vector<T> &data_vector);
    void forward_propagate_BIAS();
    void forward_propagate_NB();
    void clear_network_BIAS();
    void clear_network_NB();

    /// private member functions defined in class
    double derivative_of_sigmoid(double x){
        double num = (pow(M_E, x)) / pow((pow(M_E, x) + 1), 2);
        return num;
    }

};


#endif //NEURAL_NETWORK_GREEN_NEURAL_H
