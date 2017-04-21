
#include <cassert>
#include <math.h>
#include "Green.h"


Green::Green(int num_input_nodes, int num_output_nodes, double learning_rate, std::map<int, int> &mp) {
    neural_obj = prepare_hidden_layers(mp);
}


std::vector<std::vector<Node>> Green::prepare_hidden_layers(std::map<int, int> &mp) {

    std::vector<std::vector<Node>> master_vector;

    for (auto it = mp.begin(); it != mp.end(); ++it) {
        std::vector<Node> mv;  // layer vector
        if (it->first == 0 && mp.size() == 1) {
            for (int i = 0; i < it->second; i++) {
                Node *node = new Node(num_output_nodes, num_input_nodes);
                node->initialize_weights(num_output_nodes);     // number of weights is size of next layer over
                mv.push_back(*node);
            }
        }
        else if (it->first == 0) {
            auto itt = it; ++itt;
            for (int i = 0; i < it->second; i++) {
                Node *node = new Node(itt->second, num_input_nodes);
                node->initialize_weights(itt->second);
                mv.push_back(*node);
            }
        }
        else if (it->first == mp.size() - 1) {
            auto itt = it; --itt;
            for (int i = 0; i < it->second; i++) {
                Node *node = new Node(num_output_nodes, itt->second);
                node->initialize_weights(num_output_nodes);
                mv.push_back(*node);
            }
        }
        else {
            auto it_left = it, it_right = it; --it_left; ++it_right;
            for (int i = 0; i < it->second; i++) {
                Node *node = new Node(it_right->second, it_left->second);
                node->initialize_weights(it_right->second);
                mv.push_back(*node);
            }
        }
        master_vector.push_back(mv);
    }
    return master_vector;
}

void Green::prepare_input_layer(std::vector<std::vector<Node>> &mv) {
    std::vector<Node> input_vector;
    for (int i = 0; i < num_input_nodes; i++) {
        Node *node = new Node((int) mv[0].size(), 0);
        node->initialize_weights((int) mv[0].size());
        input_vector.push_back(*node);
    }
    auto it = mv.begin();
    mv.insert(it, input_vector);
}

void Green::prepare_output_layer(std::vector<std::vector<Node>> &mv) {
    std::vector<Node> output_vector;
    for (int i = 0; i < num_output_nodes; i++) {
        Node *node = new Node(0, (int) mv[mv.size() - 1].size());
        output_vector.push_back(*node);
    }
    mv.push_back(output_vector);
}

void Green::generate_web(std::vector<std::vector<Node>> &mv) {
    // connect forwards
    for (int i = 0; i < mv.size() - 1; i++) {
        for (int j = 0; j < mv[i].size(); j++) {
            for (int k = 0; k < mv[i + 1].size(); k++) {
                mv[i][j].attach_v_front(mv[i + 1][k]);
            }
        }
    }
    // connect backwards
    for (int p = (int) (mv.size() - 1); p >= 1; p--) {
        for (int q = 0; q < mv[p].size(); q++) {
            for (int r = 0; r < mv[p - 1].size(); r++) {
                mv[p][q].attach_v_back(mv[p - 1][r]);
            }
        }
    }
}

template <class T>
void Green::insert_data_BIAS(const std::vector<T> &data_vector) {
    assert(data_vector.size() == neural_obj[0].size() - 1);
    for (int i = 0; i < data_vector.size(); i++) {
        neural_obj[0][i].val = data_vector[i];
    }
}

template <class T>
void Green::insert_data_NB(const std::vector<T> &data_vector) {
    assert(data_vector.size() == neural_obj[0].size());
    for (int i = 0; i < data_vector.size(); i++) {
        neural_obj[0][i].val = data_vector[i];
    }
}

void Green::forward_propagate_BIAS() {
    /// begin forward pass
    for (int i = 0; i < neural_obj.size() - 1; i++) {
        for (int j = 0; j < neural_obj[i].size(); j++) {
            if (i == neural_obj.size() - 2) { // last hidden layer.
                // forward to every next node since next layer is output.
                for (int k = 0; k < neural_obj[i + 1].size(); k++) {
                    neural_obj[i + 1][k].val += (neural_obj[i][j].val * neural_obj[i][j].weights[k]);
                }
            }
            else {
                // forward to all but last node of every next layer, since we don't forward to a bias.
                for (int k = 0; k < neural_obj[i + 1].size() - 1; k++) {
                    neural_obj[i + 1][k].val += (neural_obj[i][j].val * neural_obj[i][j].weights[k]);
                }
            }
        }
        /// forwarding to next layer is complete. Now begin crush on that next layer. Sigmoid. Consider separate funct.
        if (i == neural_obj.size() - 2) { // last hidden layer.
            // crush on all next nodes since next layer is output.
            for (int p = 0; p < neural_obj[i + 1].size(); p++) {
                neural_obj[i + 1][p].val_before_sigmoid = neural_obj[i + 1][p].val;
                neural_obj[i + 1][p].val = 1 / (1 + pow(M_E, -(neural_obj[i + 1][p].val)));
            }
        }
        else {
            // crush on all but last node of evey next layer, since we don't crush on a bias.
            for (int p = 0; p < neural_obj[i + 1].size() - 1; p++) {
                neural_obj[i + 1][p].val_before_sigmoid = neural_obj[i + 1][p].val;
                neural_obj[i + 1][p].val = 1 / (1 + pow(M_E, -(neural_obj[i + 1][p].val)));
            }
        }
    }
}

void Green::forward_propagate_NB() {
    /// now begin forward pass procedures
    for (int i = 0; i < neural_obj.size() - 1; i++) { // stop at last hidden layer, since we pass forward
        for (int j = 0; j < neural_obj[i].size(); j++) {
            for (int k = 0; k < neural_obj[i + 1].size(); k++) {
                neural_obj[i + 1][k].val += (neural_obj[i][j].val * neural_obj[i][j].weights[k]);
            }
        }
        /// forwarding to next layer is complete. Now begin crush on that next layer. Sigmoid.
        for (int p = 0; p < neural_obj[i + 1].size(); p++) {
            neural_obj[i + 1][p].val_before_sigmoid = neural_obj[i + 1][p].val;
            neural_obj[i + 1][p].val = 1 / (1 + pow(M_E, -(neural_obj[i + 1][p].val)));
        }
    }
}



















