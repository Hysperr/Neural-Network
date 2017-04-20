
#ifndef NEURAL_NETWORK_NODE_H
#define NEURAL_NETWORK_NODE_H


#include <vector>
#include <cstdlib>

class Node {
public:
    /// typedefs
    typedef size_t size_type;
    /// constructor
    Node(int front_conn, int back_conn);
    /// public member functions
    size_type attach_v_front(Node &node);
    size_type attach_v_back(Node &node);
    void initialize_weights(int front_connections);

private:
    double val;
//    double val_uncrushed;
    double error;
    double real_identity;
//    int unique_id;
    int conn;
    std::vector<Node *> v_front;
    std::vector<Node *> v_back;
    std::vector<double> weights;
//    std::vector<double> weights_old;

//    friend void Green::generate_web(std::vector<std::vector<Node>> &mv);

};


#endif //NEURAL_NETWORK_NODE_H
