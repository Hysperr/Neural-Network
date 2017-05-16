
#ifndef NEURAL_NETWORK_NODE_H
#define NEURAL_NETWORK_NODE_H


#include <vector>
#include <cstdlib>
#include <map>
#include <memory>

class Node {
public:
    /*! Tightly coupled classes acting as single unit */
    friend class NeuralNet;
    typedef size_t size_type;

    /*!
     * Constructor initializes \c v_front with the
     * number of nodes in the next layer and \c v_back
     * with the number of nodes in the previous layer
     * @param front_conn - number of front connections.
     * @param back_conn - number of back connections.
     */
    Node(unsigned int front_conn, unsigned int back_conn);

    /*!
     * This function should only be called when attaching
     * to the next layer of nodes. The calling node
     * attaches to the param \c node by assigning one of
     * its pointers located within \c v_front to the param
     * \c node. One connection is made per function call.
     * If a failed attachment occurs, a message alerting
     * the user is printed to output.
     * @param node - The node to attach to.
     * @return - The number of connections made.
     */
    typename Node::size_type attach_v_front(Node &node);

    /*!
     * This function should only be called when attaching
     * to the previous layer of nodes. The calling node
     * attaches to the param \c node by assigning one of
     * its pointers located within \c v_back to the param
     * \c node. One connection is made per function call.
     * If a failed attachment occurs, a message alerting
     * the user is printed to output.
     * @param node - The node to attach to.
     * @return - The number of connections made.
     */
    typename Node::size_type attach_v_back(Node &node);

private:
    double val;                         /*!< Node's data value */

    double val_before_sigmoid;          /*!< Node's data value before processed using sigmoid function 1 / 1 + (e^-z) */

    double error;                       /*!< Error value used in data processing */

    double real_identity;               /*!< Used to identify what each output-layer node represents */

    int conn;                           /*!< Number of connections for current node. */

    std::vector<Node *> v_front;        /*!< Holds connections to every next-layer node, corresponds to \c weights field */

    std::vector<Node *> v_back;         /*!< Holds connections to every previous-layer node */

    std::vector<double> weights;        /*!< Unique floating-points for each next-layer node, corresponds to \c v_front field */

    std::vector<double> old_weights;    /*!< Only used within NeuralNet::back_propagate()


    /*!
     * Assigns arbitrarily tiny weight to each of
     * the calling node's attachments in \c v_front
     * by generating pseudorandom floating-points
     * from [0.001 - 0.0099] and pushing into node's
     * \c weights vector field. Each index within
     * \c weights corresponds to the node's attachments
     * in \c v_front. Thus each attachment has a unique
     * weight. The seed for rand() must be established
     * in driver file.
     * @param front_connections - num nodes in next layer
     * to the calling node's weight vector.
     */
    void initialize_weights(int front_connections) {
        // seed must be established in main
        for (int i = 0; i < front_connections; i++)
            weights.push_back(((double) rand() / RAND_MAX) / 100.0);
    }

};


#endif //NEURAL_NETWORK_NODE_H
