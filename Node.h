
#ifndef NEURAL_NETWORK_NODE_H
#define NEURAL_NETWORK_NODE_H


#include <vector>
#include <cstdlib>
#include <map>
#include <memory>

class Node {
public:
    friend class Green;     // tightly coupled classes acting as one unit
    /// typedefs
    typedef size_t size_type;
    /// constructor
    Node(int front_conn, int back_conn);
    /// public member functions
    size_type attach_v_front(Node &node);
    size_type attach_v_back(Node &node);
    void set_output_identity(std::vector<std::vector<Node>> &mv, const std::map<int, double> &identity_map);


private:
    double val;
    double val_before_sigmoid;
    double error;
    double real_identity;
//    int unique_id;
    int conn;
    std::vector<Node *> v_front;
    std::vector<Node *> v_back;
    std::vector<double> weights;
//    std::vector<double> weights_old;

    void initialize_weights(int front_connections) {
        /// seed must be established in main
        for (int i = 0; i < front_connections; i++) {
            weights.push_back( (double) rand() / RAND_MAX / 10);
        }
    }


};


#endif //NEURAL_NETWORK_NODE_H