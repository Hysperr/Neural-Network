
#ifndef NEURAL_NETWORK_GREEN_NEURAL_H
#define NEURAL_NETWORK_GREEN_NEURAL_H


#include "Node.h"
#include <map>
#include <cmath>

class NeuralNet {
public:
    // ctors
    /*!
     * Constructor initializing a neural net object backed by a 2D vector.
     * @param num_input_nodes - The number of input nodes to generate.
     * @param num_output_nodes - The number of output nodes to generate.
     * @param learning_rate - The learning rate/alpha.
     * @param mp - The map - The \c std::map used to generate the hidden layers.
     * @param include_bias - determines inclusion of bias nodes - all layers except output-layer
     */
    NeuralNet(unsigned int num_input_nodes, unsigned int num_output_nodes, double learning_rate,
              std::map<unsigned, unsigned> &mp,
              bool include_bias = false);

    // getters
    /*! Returns constant reference to the read-only backed 2D vector (net) */
    const std::vector<std::vector<Node>> &getNeural_obj() const { return mv; }

    typename Node::size_type get_total_connections() const { return total_conn; }

    int get_num_output_nodes() const { return num_output_nodes; }

    int get_num_input_nodes() const { return num_input_nodes; }

    double get_learning_rate() const { return learning_rate; }

    bool biasIsActive() const { return bias; }

    // operations
    /*!
     * Sets \c real_identity field within output-layer nodes
     * using \c std::map. key/value pairs key = index (must
     * start from 0, increment by 1), value = \c real identity.
     * An assertion ensures the map's size is the same size as
     * the number of output nodes. The index of each element
     * within the map corresponds to the index of each output
     * layer node. Default parameter \c debug_print = false.
     * True outputs each output-layer's real_identity.
     * @param identity_map
     * @param debug_print
     */
    void set_output_identity(const std::map<unsigned, double> &identity_map, bool debug_print = false);

    /*!
     * Public function call for user which calls 1 of 2
     * private forward propagate functions to account for
     * whether bias nodes are active.
     * f..._BIAS is called when \c bias = true
     * f..._NB is called when \c bias = false
     */
    void forward_propagate() { (bias) ? forward_propagate_BIAS() : forward_propagate_NB(); }

    /*!
     * Public function call for user which calls 1 of 2
     * private clear network functions to account for
     * whether bias nodes are active.
     * clear_n..._BIAS is called when \c bias = true
     * clear_n..._NB is called when \c bias = false
     */
    void clear_network() { (bias) ? clear_network_BIAS() : clear_network_NB(); }

    /*!
     * Neural net decides what the input data it has
     * processed actually represents. Works by selecting
     * the highest value output node after forward propagation
     * (determined by \c val ). If that selected node's identity,
     * set earlier by \c set_output_identity(), corresponds to
     * the \c label parameter, then the neural net's decision
     * is correct and \c true is returned. Default parameter
     * debug_print = false. True outputs ALL of neural net's data
     * used to make its decision along with a visual illustration
     * of where it guessed correctly and incorrectly.
     *
     * @param label - The real identity of the input data.
     * @param debug_print - Default parameter. True prints belief data.
     * @return
     */
    bool choose_answer(const double label, bool debug_print = false) const;

    /*!
     * Inserts raw data into neural net's input-layer \c mv[0].
     * An assertion checks the size of \c data_vector is equal to
     * the number of input nodes.
     * @param data_vector
     */
    void insert_data(const std::vector<double> &data_vector);

    /*!
     * Back propagates corrective reinforcement behavior backwards
     * through neural net by adjusting weight values of each node.
     * In supervised reinforcement practice this is performed many
     * times until a "fine-tuning" is achieved among the weights.
     * Works by first copying \c weights into \c old_weights then...
     * 1) Calculate the errors of output neurons.
     * 2) Change the output layer's incoming weights.
     * 3) Calculate all hidden errors.
     * 4) Change hidden layer weights.
     * @param label - The real identity of the input data.
     */
    void back_propagate(const double label);

    // prints

    void print_neural_layer(int index) const;

    void print_output_layer() const;

    void print_input_layer() const;

    void print_ENTIRE_network() const;

private:
    bool bias;                              /*!< Determines if bias nodes are active for this neural net */

    unsigned int num_input_nodes;           /*!< Number of input nodes */

    unsigned int num_output_nodes;          /*!< Number of output nodes */

    double learning_rate;                   /*!< Learning rate/alpha */

    Node::size_type total_conn;             /*!< Total connections in neural net, updated by Node::attach_v_front() */

    std::vector<std::vector<Node>> mv;      /*!< 2D vector backing neural net object. Each index is a layer */


    /*!
     * The \c std::map used to generate hidden layers of
     * neural net. key = layer number (must start from 0
     * and increment by 1), value = num nodes per layer.
     * @param mp - The map used to generate hidden layers
     * @return - Returns 2D vector
     */
    std::vector<std::vector<Node>> prepare_hidden_layers(std::map<unsigned, unsigned> &mp);

    /*!
     * Generates input layer of neural net by inserting
     * \c std::vector<Node> at beginning of \c mv
     * postcondition: mv[0] is hidden layer and
     * mv[1] though mv[mv.size() - 1] are hidden layers.
     */
    void prepare_input_layer();

    /*!
     * Generates output layer of neural net by inserting
     * \c std::vector<Node> at end of \c mv
     * postcondition: mv[mv.size() - 1] is output layer.
     */
    void prepare_output_layer();

    /*!
     * Generates connections between nodes throughout entire
     * neural net forwards and backwards. Works by calling
     * \c Node::attach_v_front and \c Node::attach_v_back
     * for each node. For example, current node of current
     * layer attaches to all nodes of next layer. Number of
     * nodes in next or previous layers denoted by \c k
     * within definition's for-loops. \c total_conn field
     * is updated ONLY within connect forwards section!
     * because logically a single connection is needed
     * between any two nodes. The back connections are
     * added to simplify algorithms where needed.
     */
    void generate_neural_web();

    /*!
     * Generates bias nodes on neural network. Bias nodes
     * are located on every layer except the output layer.
     * Bias nodes are placed at the bottom (end of vector)
     * of each layer. Bias nodes always have a value of 1.
     * Bias nodes attach to the next layer only, never the
     * previous layer. Bias nodes forward propagate and have
     * unique weights like other nodes. Bias node's weights
     * are updated during backpropagation like other nodes
     * but again, their value always remains 1. Bias nodes
     * allow curve shifting for better fit.
     */
    void generate_bias_nodes();

    // Below, their public counterparts decide which to call

    /*!
     * precondition: <code> bias = true </code>
     * This function takes into account bias nodes by not
     * forwarding to them (since they do not receive and
     * do not have any previous-layer connections).
     * Forward propagates input data from input layer through
     * neural network to the output layer. How it works:
     * For each next layer node, each current layer node
     * multiplies its value by its weight that corresponds
     * to the next layer's node and places that value in
     * the connecting node. After forwarding for each layer
     * is complete, the values in the receiving layer nodes
     * are "crushed" using sigmoid function 1 / (1 + (e^-z)
     * where z is the value the node just received from the
     * previous layer. All layers except output layer forward
     * propagates. Crushing does not occur on the output layer.
     * Remember, crushing happens on the next layer once
     * forwarding from the previous layer is complete!
     */
    void forward_propagate_BIAS();

    /*!
     * precondition: <code> bias = false </code>
     * This function does not take into account bias nodes.
     * Forward propagates input data from input layer through
     * neural network to the output layer. How it works:
     * For each next layer node, each current layer node
     * multiplies its value by its weight that corresponds
     * to the next layer's node and places that value in
     * the connecting node. After forwarding for each layer
     * is complete, the values in the receiving layer nodes
     * are "crushed" using sigmoid function 1 / (1 + (e^-z)
     * where z is the value the node just received from the
     * previous layer. All layers except output layer forward
     * propagates. Crushing does not occur on the output layer.
     * Remember, crushing happens on the next layer once
     * forwarding from the previous layer is complete!
     */
    void forward_propagate_NB();

    /*!
     * precondition: <code> bias = true </code>
     * postcondition: \c val and \c val_before_sigmoid
     * are reset to 0
     */
    void clear_network_BIAS();

    /*!
     * precondition: <code> bias = false </code>
     * postcondition: \c val and \c val_before_sigmoid
     * are reset to 0
     */
    void clear_network_NB();

    /*!
     * Calculates derivative of \c val_before_sigmoid within back_propagate()
     * @param x - \c value_before_sigmoid.
     * @return - The derivative.
     */
    double derivative_of_sigmoid(double x) {
        return (pow(M_E, x)) / pow((pow(M_E, x) + 1), 2);
    }

};


#endif //NEURAL_NETWORK_GREEN_NEURAL_H
