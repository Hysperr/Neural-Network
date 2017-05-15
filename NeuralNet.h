
#ifndef NEURAL_NETWORK_GREEN_NEURAL_H
#define NEURAL_NETWORK_GREEN_NEURAL_H


#include "Node.h"
#include <map>
#include <cmath>

class NeuralNet {
public:
    // ctors
    /*!
     * Constructor initializing a neural net object.
     * @param num_input_nodes - The number of input nodes to generate.
     * @param num_output_nodes - The number of output nodes to generate.
     * @param learning_rate - The learning rate/alpha.
     * @param mp - The map - The \c std::map used to generate the hidden layers.
     * @param include_bias - Boolean that determines whether to insert bias nodes - all layers except output-layer
     */
    NeuralNet(unsigned int num_input_nodes, unsigned int num_output_nodes, double learning_rate,
              std::map<unsigned, unsigned> &mp,
              bool include_bias = false);

    // getters
    /*! Returns constant reference to the neural net. Neural net is read-only; cannot be modified! */
    const std::vector<std::vector<Node>> &getNeural_obj() const { return mv; }

    typename Node::size_type get_total_connections() const { return total_conn; }

    int getNum_output_nodes() const { return num_output_nodes; }

    int getNum_input_nodes() const { return num_input_nodes; }

    double getLearning_rate() const { return learning_rate; }

    bool biasIsActive() const { return bias; }

    // operations
    /*! Sets \c real_identity field within output-layer nodes.
     * Map key = index (must start from 0!), value = \c real_identity.
     * An assertion ensures the map's size is the same size as the number
     * of output nodes. The index of each element within the map corresponds
     * to the index of each output layer node. Default parameter \c debug_print = false.
     * true outputs each output-layer's real_identity.
     * @param identity_map
     * @param debug_print
     */
    void set_output_identity(const std::map<unsigned, double> &identity_map, bool debug_print = false);

    /*! Public interface to call 1 of 2 private forward propagate functions.
     * Whether bias is active determines the appropriate call.
     */
    void forward_propagate() { (bias) ? forward_propagate_BIAS() : forward_propagate_NB(); }

    /*! Public interface to call 1 of 2 \c private clear_network functions.
     * Whether bias is active determines the appropriate call.
     */
    void clear_network() { (bias) ? clear_network_BIAS() : clear_network_NB(); }

    /*! Constant member function. Neural net makes its decision on what the input data
     * really is. It works by selecting the highest value output node (determined by \c val )
     * If that node's identity, set earlier by \c set_output_identity(), corresponds to the
     * \c label parameter, then the neural net's decision is correct and \c true is returned.
     * Default parameter debug_print = false. true outputs ALL of neural net's data it used to
     * make its decision and what it believes the input data is, along with a visual representation
     * of where it guessed correctly and incorrectly.
     *
     * @param label - The value the input data represents for current epoch.
     * @param debug_print - Default parameter false by default. True prints neural net's belief data.
     * @return
     */
    bool choose_answer(const double label, bool debug_print = false) const;

    /*! Inserts raw data into neural net's input-layer.
     * An assertion checks the size of \c data_vector is equal to
     * the number of input nodes.
     * @param data_vector
     */
    void insert_data(const std::vector<double> &data_vector);

    /*! Back propagates corrective reinforcement behavior backwards through neural net
     * by adjusting weight values of each node. In supervised reinforcement practice this is performed many times until
     * a "fine-tuning" is achieved among the weights. It works by copying the \c weights vector into \c old_weights vector
     * 1) Calculate the errors of output neurons.
     * 2) Change the output layer's incoming weights.
     * 3) Calculate all hidden errors.
     * 4) Change hidden layer weights.
     * @param label - The value the input data represents for current epoch.
     */
    void back_propagate(const double label);

    // prints
    void print_neural_layer(int index) const;

    void print_ENTIRE_network() const;

    void print_output_layer() const;

    void print_input_layer() const;

private:
    bool bias;
    unsigned int num_input_nodes;
    unsigned int num_output_nodes;
    double learning_rate;
    Node::size_type total_conn;
    std::vector<std::vector<Node>> mv;


    // private member functions (we use these to build the neural object)
    std::vector<std::vector<Node>> prepare_hidden_layers(std::map<unsigned, unsigned> &mp);

    void prepare_input_layer();

    void prepare_output_layer();

    void generate_neural_web();

    void generate_bias_nodes();

    // their public counterparts decide which to call NB = no bias nodes, BIAS = bias nodes
    void forward_propagate_BIAS();

    void forward_propagate_NB();

    void clear_network_BIAS();

    void clear_network_NB();

    // private member function inlined in class
    double derivative_of_sigmoid(double x) {
        return (pow(M_E, x)) / pow((pow(M_E, x) + 1), 2);
    }

};


#endif //NEURAL_NETWORK_GREEN_NEURAL_H
