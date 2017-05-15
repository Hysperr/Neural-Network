
#ifndef NEURAL_NETWORK_NODE_H
#define NEURAL_NETWORK_NODE_H


#include <vector>
#include <cstdlib>
#include <map>
#include <memory>

class Node {
public:
    friend class NeuralNet;             /*!< Tightly coupled classes acting as single unit */
    typedef size_t size_type;

    /*! Constructor
     * Used to initialize v_front with
     * the number of nodes in the next-layer and
     * v_back with the number of nodes in the previous-layer
     * @param front_conn
     * @param back_conn
     */
    Node(unsigned int front_conn, unsigned int back_conn);

    /*! Attaches \c this node to \c node.
     * This function should only be called
     * when attaching to the next layer of nodes.
     * The calling node attaches to the param \c node by
     * assigning one of its pointers located within v_front to
     * the parameter \c node. One connection
     * will be made per function call. If a failed attachment occurs,
     * a message alerting the user is printed to output.
     * @param node - The node to attach to.
     * @return - The number of connections made.
     */
    typename Node::size_type attach_v_front(Node &node);

    /*! Attaches \c this node to \c node.
     * This function should only be called
     * when attaching to the previous layer of nodes.
     * The calling node attaches to the param \c node by
     * assigning one of its pointers located within v_back to
     * the parameter \c node. One connection
     * will be made per function call. If a failed attachment occurs,
     * a message alerting the user is printed to output.
     * @param node - The node to attach to.
     * @return - The number of connections made.
     */
    typename Node::size_type attach_v_back(Node &node);

private:
    double val;
    double val_before_sigmoid;
    double error;
    double real_identity;               /*!< Used to identify what each output-layer node represents */
    int conn;
    std::vector<Node *> v_front;        /*!< Holds connections to every next-layer node, corresponds to \c weights field */
    std::vector<Node *> v_back;         /*!< Holds connections to every previous-layer node */
    std::vector<double> weights;        /*!< Unique floating-points for each next-layer node, corresponds to \c v_front field */
    std::vector<double> old_weights;    /*!< Only used within NeuralNet::back_propagate()


    /*! Assigns arbitrarily small weights to calling node's connections.
     * Very small floating points are assigned to the calling node's weight vector.
     * It's weight vector correspond to its physical node attachments in v_front.
     * Therefore each connection has a unique, small weight. The seed must be
     * established in driver file.
     * @param front_connections - Determines the number of values to assign
     * to the calling node's weight vector.
     */
    void initialize_weights(int front_connections) {
        // seed must be established in main
        for (int i = 0; i < front_connections; i++)
            weights.push_back(((double) rand() / RAND_MAX) / 100.0);
    }

};


#endif //NEURAL_NETWORK_NODE_H
