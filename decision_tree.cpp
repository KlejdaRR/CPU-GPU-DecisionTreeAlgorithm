#include "decision_tree.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <iostream>

// Constructor
DecisionTree::DecisionTree(int max_depth) : max_depth(max_depth), root(nullptr) {}

// Destructor
DecisionTree::~DecisionTree() {
    delete_tree(root);
}

// Helper function to delete the tree
void DecisionTree::delete_tree(Node* node) {
    if (node) {
        delete_tree(node->left);
        delete_tree(node->right);
        delete node;
    }
}

// Calculate Gini impurity
double DecisionTree::calculate_gini(const std::vector<int>& labels) {
    if (labels.empty()) return 0.0;

    std::vector<int> counts(3, 0); // Assuming 3 classes
    for (int label : labels) {
        counts[label]++;
    }

    double gini = 1.0;
    for (int count : counts) {
        double probability = static_cast<double>(count) / labels.size();
        gini -= probability * probability;
    }

    return gini;
}

// Build the decision tree
Node* DecisionTree::build_tree(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int depth) {
    int num_samples = data.size();
    int num_features = data[0].size();

    std::cout << "Depth: " << depth << ", Samples: " << num_samples << std::endl;

    // Stopping conditions
    bool all_same = std::all_of(labels.begin(), labels.end(), [&](int v) { return v == labels[0]; });
    if (depth >= max_depth || num_samples <= 2 || all_same) {
        Node* leaf = new Node();
        leaf->value = std::distance(labels.begin(), std::max_element(labels.begin(), labels.end()));
        leaf->left = nullptr;
        leaf->right = nullptr;
        std::cout << "Created leaf node with value: " << leaf->value << std::endl;
        return leaf;
    }

    double best_gini = 1.0;
    int best_feature = -1;
    double best_threshold = 0.0;

    #pragma omp parallel for collapse(2)
    for (int feature_index = 0; feature_index < num_features; feature_index++) {
        for (int sample_index = 0; sample_index < num_samples; sample_index++) {
            double threshold = data[sample_index][feature_index];

            std::vector<int> left_labels, right_labels;
            for (int i = 0; i < num_samples; i++) {
                if (data[i][feature_index] <= threshold) {
                    left_labels.push_back(labels[i]);
                } else {
                    right_labels.push_back(labels[i]);
                }
            }

            double gini_left = calculate_gini(left_labels);
            double gini_right = calculate_gini(right_labels);
            double weighted_gini = (left_labels.size() * gini_left + right_labels.size() * gini_right) / num_samples;

            #pragma omp critical
            {
                if (weighted_gini < best_gini) {
                    best_gini = weighted_gini;
                    best_feature = feature_index;
                    best_threshold = threshold;
                }
            }
        }
    }

    std::cout << "Best feature: " << best_feature << ", Best threshold: " << best_threshold << std::endl;

    Node* node = new Node();
    node->feature_index = best_feature;
    node->threshold = best_threshold;

    std::vector<std::vector<double>> left_data, right_data;
    std::vector<int> left_labels, right_labels;
    for (int i = 0; i < num_samples; i++) {
        if (data[i][best_feature] <= best_threshold) {
            left_data.push_back(data[i]);
            left_labels.push_back(labels[i]);
        } else {
            right_data.push_back(data[i]);
            right_labels.push_back(labels[i]);
        }
    }

    node->left = build_tree(left_data, left_labels, depth + 1);
    node->right = build_tree(right_data, right_labels, depth + 1);

    return node;
}

// Fit the decision tree
void DecisionTree::fit(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
    std::cout << "Starting to build the decision tree..." << std::endl;
    root = build_tree(data, labels, 0);
    std::cout << "Decision tree built successfully." << std::endl;
}

// Predict using the decision tree
int DecisionTree::predict(const std::vector<double>& sample) {
    std::cout << "Predicting sample..." << std::endl;
    Node* node = root;
    while (node->left || node->right) {
        if (sample[node->feature_index] <= node->threshold) {
            node = node->left;
        } else {
            node = node->right;
        }
    }
    std::cout << "Predicted value: " << node->value << std::endl;
    return node->value;
}
