
#ifndef NEURAL_NETWORK_GREEN_NEURAL_H
#define NEURAL_NETWORK_GREEN_NEURAL_H


#include "Node.h"
#include <map>


class Green {
//    friend class Node;
    /// public members
public:
    /// constructors
    Green(int num_input_nodes, int num_output_nodes, double learning_rate, std::map<int, int> &mp);
    /// public member functions
    int getNum_input_nodes() const { return num_input_nodes; }
    int getNum_output_nodes() const { return num_output_nodes; }
    double getLearning_rate() const { return learning_rate; }
    template <class T> void insert_data_BIAS(const std::vector<T> &data_vector);
    template <class T> void insert_data_NB(const std::vector<T> &data_vector);
    void forward_propagate_BIAS();
    void forward_propagate_NB();


private:
/// private members
    int num_input_nodes;
    int num_output_nodes;
    double learning_rate;
    Node::size_type total_attachments_made;
    std::vector<std::vector<Node>> neural_obj;

    /// private member functions (we use these to build the neural object)
    std::vector<std::vector<Node>> prepare_hidden_layers(std::map<int, int>& mp);
    void prepare_input_layer(std::vector<std::vector<Node>> &mv);
    void prepare_output_layer(std::vector<std::vector<Node>> &mv);
    void generate_web(std::vector<std::vector<Node>> &mv);

};


#endif //NEURAL_NETWORK_GREEN_NEURAL_H
