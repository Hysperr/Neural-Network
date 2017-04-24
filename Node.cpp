
#include "Node.h"
#include "NeuralNet.h"
#include <iostream>

Node::Node(unsigned int front_conn, unsigned int back_conn) : val(0),
                                                              val_before_sigmoid(0),
                                                              real_identity(-1),
                                                              error(0),
                                                              conn(0),
                                                              weights(),
                                                              old_weights(),
                                                              v_front(front_conn, nullptr),
                                                              v_back(back_conn, nullptr) {
}

Node::size_type Node::attach_v_front(Node &node) {
    bool flag = false;
    size_type num_attaches = 0;
    for (int i = 0; i < v_front.size(); i++) {
        if (v_front[i] == (void *) 0) {
            v_front[i] = &node;
            conn++;
            num_attaches++;
            flag = true;
            break;
        }
    }
    if (!flag) {
        std::cout << "ATTACHMENT FAILED - FRONT\n";
    }
    return num_attaches;
}

Node::size_type Node::attach_v_back(Node &node) {
    bool flag = false;
    size_type num_attaches = 0;
    for (int i = 0; i < v_back.size(); i++) {
        if (v_back[i] == (void *) 0) {
            v_back[i] = &node;
            conn++;
            num_attaches++;
            flag = true;
            break;
        }
    }
    if (!flag) {
        std::cout << "ATTACHMENT FAILED - BACK\n";
    }
    return num_attaches;
}
