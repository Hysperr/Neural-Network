
#include "Node.h"
#include "Green.h"
#include <iostream>
#include <cassert>
#include <bits/unique_ptr.h>

Node::Node(int front_conn, int back_conn) : val(0), error(0), real_identity(-1), conn(0), weights(), v_front(
        (unsigned long long int) front_conn, nullptr), v_back((unsigned long long int) back_conn, nullptr) {

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
        // give the node id that it failed on
    }
    return num_attaches;
    // consider returning the node, making this a friend function of neural class that updates total attaches variable
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
        // give the node id that it failed on
    }
    return num_attaches;
    // consider returning the node, making this a friend function of neural class that updates total attaches variable
}

void Node::set_output_identity(std::vector<std::vector<Node>> &mv, const std::map<int, double> &identity_map) {
    assert(identity_map.size() == mv[mv.size() - 1].size());
    auto it = identity_map.begin();
    for (int i = 0; i < mv[mv.size() - 1].size(); i++) {
        mv[mv.size() - 1][i].real_identity = it->second;
        // std::cout << "output node " << i << " has real_identity " << it->second << '\n';       // debug
        ++it;
    }
}



