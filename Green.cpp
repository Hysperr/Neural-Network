
#include <cassert>
#include "Green.h"
#include <iostream>
#include <algorithm>

Green::Green(int num_input_nodes, int num_output_nodes, double learning_rate, std::map<int, int> &mp, bool include_bias) {
    this->num_input_nodes = num_input_nodes;
    this->num_output_nodes = num_output_nodes;
    this->learning_rate = learning_rate;
    this->bias = include_bias;
    mv = prepare_hidden_layers(mp);
    prepare_input_layer(mv);
    prepare_output_layer(mv);
    generate_neural_web(mv);
    if (include_bias)
        generate_bias_nodes();
}


std::vector<std::vector<Node>> Green::prepare_hidden_layers(std::map<int, int> &mp) {
    std::vector<std::vector<Node>> master_vector;
    for (auto it = mp.begin(); it != mp.end(); ++it) {
        std::vector<Node> layer;  // layer vector
        if (it->first == 0 && mp.size() == 1) {
            for (int i = 0; i < it->second; i++) {
                Node *node = new Node(num_output_nodes, num_input_nodes);
                node->initialize_weights(num_output_nodes);     // number of weights is size of next layer over
                layer.push_back(*node);
            }
        }
        else if (it->first == 0) {
            auto itt = it; ++itt;
            for (int i = 0; i < it->second; i++) {
                Node *node = new Node(itt->second, num_input_nodes);
                node->initialize_weights(itt->second);
                layer.push_back(*node);
            }
        }
        else if (it->first == mp.size() - 1) {
            auto itt = it; --itt;
            for (int i = 0; i < it->second; i++) {
                Node *node = new Node(num_output_nodes, itt->second);
                node->initialize_weights(num_output_nodes);
                layer.push_back(*node);
            }
        }
        else {
            auto it_left = it, it_right = it; --it_left; ++it_right;
            for (int i = 0; i < it->second; i++) {
                Node *node = new Node(it_right->second, it_left->second);
                node->initialize_weights(it_right->second);
                layer.push_back(*node);
            }
        }
        master_vector.push_back(layer);
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

void Green::generate_neural_web(std::vector<std::vector<Node>> &mv) {
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

void Green::generate_bias_nodes() {
    for (int i = 0; i < (int) mv.size() - 1; i++) {
        Node *bias_node = new Node((int) mv[i + 1].size(), 0);
        bias_node->val = 1;
        bias_node->initialize_weights((int) mv[i + 1].size());
        mv[i].push_back(*bias_node);
        for (int u = 0; u < (int) mv[i + 1].size(); u++) {
            mv[i][mv[i].size() - 1].attach_v_front(mv[i + 1][u]);
        }
    }
}


void Green::insert_data(const std::vector<double> &data_vector) {
    assert(data_vector.size() == num_input_nodes);
    for (int i = 0; i < data_vector.size(); i++) {
        mv[0][i].val = data_vector[i];
    }
}

// make value a double in the mapping
void Green::set_output_identity(const std::map<int, int> &identity_map, bool debug_print) {
    assert(identity_map.size() == num_output_nodes);
    std::map<int, int>::const_iterator it = identity_map.begin();   // or use auto for clarify
    for (int i = 0; i < identity_map.size(); i++) {
        mv[mv.size() - 1][i].real_identity = it->second;
        if (debug_print)
            std::cout << "output node " << i << " has real_identity " << it->second << '\n';
        ++it;
    }
}

void Green::forward_propagate_BIAS() {
    /// begin forward pass
    for (int i = 0; i < mv.size() - 1; i++) {
        for (int j = 0; j < mv[i].size(); j++) {
            if (i == mv.size() - 2) { // last hidden layer.
                for (int k = 0; k < mv[i + 1].size(); k++) {
                    mv[i + 1][k].val += (mv[i][j].val * mv[i][j].weights[k]);
                }
            }
            else {
                for (int k = 0; k < mv[i + 1].size() - 1; k++) {
                    mv[i + 1][k].val += (mv[i][j].val * mv[i][j].weights[k]);
                }
            }
        }
        /// forwarding to next layer is complete. Now begin crush on that next layer. Sigmoid. Consider separate funct.
        if (i == mv.size() - 2) { // last hidden layer.
            for (int p = 0; p < mv[i + 1].size(); p++) {
                mv[i + 1][p].val_before_sigmoid = mv[i + 1][p].val;     // special case
                mv[i + 1][p].val = 1 / (1 + pow(M_E, -(mv[i + 1][p].val)));
            }
        }
        else {
            for (int p = 0; p < mv[i + 1].size() - 1; p++) {
                mv[i + 1][p].val_before_sigmoid = mv[i + 1][p].val;     // special case
                mv[i + 1][p].val = 1 / (1 + pow(M_E, -(mv[i + 1][p].val)));
            }
        }
    }
}

void Green::forward_propagate_NB() {
    /// now begin forward pass procedures
    for (int i = 0; i < mv.size() - 1; i++) {
        for (int j = 0; j < mv[i].size(); j++) {
            for (int k = 0; k < mv[i + 1].size(); k++) {
                mv[i + 1][k].val += (mv[i][j].val * mv[i][j].weights[k]);
            }
        }
        /// forwarding to next layer is complete. Now begin crush on that next layer. Sigmoid.
        for (int p = 0; p < mv[i + 1].size(); p++) {
            mv[i + 1][p].val_before_sigmoid = mv[i + 1][p].val;
            mv[i + 1][p].val = 1 / (1 + pow(M_E, -(mv[i + 1][p].val)));
        }
    }
}

void Green::back_propagate(const double &label) {
/// DOES NOT TAKE BIAS NODE INTO CONSIDERATION ///
    for (int i = 0; i < mv.size(); i++) {
        for (int j = 0; j < mv[i].size(); j++) {
            mv[i][j].old_weights = mv[i][j].weights;
        }
    }
    /// 1) calculate errors of output neurons
    for (int i = 0; i < mv[mv.size() - 1].size(); i++) {
        if (label == mv[mv.size() - 1][i].real_identity)    // in case output is fraction, use the real identity, not i
            mv[mv.size() - 1][i].val_before_sigmoid = derivative_of_sigmoid(mv[mv.size() - 1][i].val_before_sigmoid) * (1 - mv[mv.size() - 1][i].val);
        else
            mv[mv.size() - 1][i].val_before_sigmoid = derivative_of_sigmoid(mv[mv.size() - 1][i].val_before_sigmoid) * (0 - mv[mv.size() - 1][i].val);
        mv[mv.size() - 1][i].error = mv[mv.size() - 1][i].val_before_sigmoid;
    }
    /// 2) change output layer's incoming weights
    for (int i = 0; i < mv[mv.size() - 2].size(); i++) {
        for (int j = 0; j < mv[mv.size() - 1].size(); j++) {
            mv[mv.size() - 2][i].weights[j] = mv[mv.size() - 2][i].weights[j] + learning_rate * mv[mv.size() - 1][j].error * mv[mv.size() - 2][i].val;
        }
    }
    /// 3) calculate all hidden errors
    for (int i = (int) mv.size() - 2; i >= 1; i--) {
        for (int j = 0; j < mv[i].size(); j++) {
            double err_gather = 0;
            for (int k = 0; k < mv[i][j].v_front.size(); k++) {
                err_gather += (mv[i][j].old_weights[k] * mv[i][j].v_front[k]->error);
            }
            mv[i][j].error = derivative_of_sigmoid(mv[i][j].val_before_sigmoid) * err_gather;
            mv[i][j].val_before_sigmoid = mv[i][j].error;   // for OOOO
        }
    }
    /// 4) change hidden layer weights
    for (int i = (int) (mv.size() - 3); i >= 0; i--) {
        for (int j = 0; j < mv[i].size(); j++) {
            for (int k = 0; k < mv[i][j].v_front.size(); k++) {
                mv[i][j].weights[k] = mv[i][j].weights[k] + (learning_rate * mv[i][j].v_front[k]->error * mv[i][j].val);
            }
        }
    }
}

void Green::clear_network_BIAS() {
    for (int i = 1; i < mv.size(); i++) {
        if (i == mv.size() - 1) {
            for (int j = 0; j < mv[i].size(); j++) {
                mv[i][j].val = 0;
                mv[i][j].val_before_sigmoid = 0;
            }
        }
        else {
            for (int j = 0; j < mv[i].size() - 1; j++) {
                mv[i][j].val = 0;
                mv[i][j].val_before_sigmoid = 0;
            }
        }
    }
}

void Green::clear_network_NB() {
    for (int i = 1; i < mv.size(); i++) {
        for (int j = 0; j < mv[i].size(); j++) {
            mv[i][j].val = 0;
            mv[i][j].val_before_sigmoid = 0;
        }
    }
}

void Green::print_neural_layer(int index) const {
    std::vector<Node> layer = mv[index];
    if (index == mv.size() - 1)
        std::cout << "========== OUTPUT LAYER (INDEX " << index << ") ==========" << std::endl;
    else if (index == 0)
        std::cout << "========== INPUT LAYER (INDEX " << index << ") ==========" << std::endl;
    else
        std::cout << "========== HIDDEN LAYER (INDEX " << index << " )========== " << std::endl;
    for (Node node : layer) {
        std::cout << "val " << node.val << '\n';
        std::cout << "val_before_sigmoid " << node.val_before_sigmoid << '\n';
        std::cout << "conn " << node.conn << '\n';
//        std::cout << "id# " << node.unique_id << '\n';
        std::cout << "identity " << node.real_identity << '\n';
        std::cout << "weights "; for (double num : node.weights) std::cout << num << ' '; std::cout << '\n';
        std::cout << '\n';
    }
    std::cout << "TOTAL " << layer.size() << " NODES IN LAYER " << index << '\n';
}

void Green::print_output_layer() const {
    int index = (int) mv.size() - 1;
    std::cout << "========== OUTPUT LAYER (INDEX " << index << ") ==========" << std::endl;
    std::vector<Node> ovec = mv[mv.size() - 1];
    for (Node node : ovec) {
        std::cout << "val " << node.val << '\n';
        std::cout << "val_before_sigmoid " << node.val_before_sigmoid << '\n';
        std::cout << "conn " << node.conn << '\n';
//        std::cout << "id# " << node.unique_id << '\n';
        std::cout << "identity " << node.real_identity << '\n';
        std::cout << "weights "; for (double num : node.weights) std::cout << num << ' '; std::cout << '\n';
        std::cout << '\n';
    }
}

void Green::print_ENTIRE_network() const {
    for (int i = 0; i < mv.size(); i++) {
        if (i == mv.size() - 1)
            std::cout << "========== OUTPUT LAYER (INDEX " << i << ") ==========" << std::endl;
        else if (i == 0)
            std::cout << "========== INPUT LAYER (INDEX " << i << ") ==========" << std::endl;
        else
            std::cout << "========== HIDDEN LAYER (INDEX " << i << " )========== " << std::endl;
        std::vector<Node> layer = mv[i];
        for (Node node : layer) {
            std::cout << "val: " << node.val << '\n';
            std::cout << "val_before_sigmoid " << node.val_before_sigmoid << '\n';
            std::cout << "conn: " << node.conn << '\n';
            std::cout << "identity: " << node.real_identity << '\n';
            std::cout << "weights: "; for (double w : node.weights) std::cout << w << ' ';
            std::cout << "\n\n";
        }
    }
}

bool Green::choose_answer(const double &label, bool debug_print) const {
    std::vector<double> max_vector;
    for (int i = 0; i < mv[mv.size() - 1].size(); i++) {
        max_vector.push_back(mv[mv.size() - 1][i].val);
    }
    std::vector<double>::iterator answer_iter = std::max_element(max_vector.begin(), max_vector.end());
    double pos = (int) (answer_iter - max_vector.begin());
    if (debug_print) {
        std::cout << "Based on my output value of " << *answer_iter
                  << " I believe this is a(n) " << mv[mv.size() - 1][pos].real_identity << '\n';
        std::cout << "In reality this is a(n) " << label << '\n';
    }
    return (mv[mv.size() - 1][pos].real_identity == label);
}

void Green::print_input_layer() const {
    std::cout << "========== INPUT LAYER (INDEX 0) ==========" << std::endl;
    std::vector<Node> ovec = mv[0];
    for (Node node : ovec) {
        std::cout << "val " << node.val << '\n';
        std::cout << "val_before_sigmoid " << node.val_before_sigmoid << '\n';
        std::cout << "conn " << node.conn << '\n';
//        std::cout << "id# " << node.unique_id << '\n';
        std::cout << "identity " << node.real_identity << '\n';
        std::cout << "weights "; for (double num : node.weights) std::cout << num << " "; std::cout << '\n';
        std::cout << '\n';
    }
}
