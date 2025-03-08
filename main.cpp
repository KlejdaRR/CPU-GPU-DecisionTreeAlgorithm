#include <iostream>
#include <vector>
#include <random>
#include "data_loader.h"
#include "decision_tree.h"
#include <algorithm>

// Function to split the dataset into training and testing sets
void split_dataset(const std::vector<std::vector<double>>& data, const std::vector<int>& labels,
                   std::vector<std::vector<double>>& train_data, std::vector<int>& train_labels,
                   std::vector<std::vector<double>>& test_data, std::vector<int>& test_labels,
                   double train_ratio = 0.8) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);

    size_t train_size = static_cast<size_t>(data.size() * train_ratio);

    for (size_t i = 0; i < indices.size(); ++i) {
        if (i < train_size) {
            train_data.push_back(data[indices[i]]);
            train_labels.push_back(labels[indices[i]]);
        } else {
            test_data.push_back(data[indices[i]]);
            test_labels.push_back(labels[indices[i]]);
        }
    }
}

// Function to calculate accuracy
double calculate_accuracy(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels) {
    if (true_labels.size() != predicted_labels.size()) {
        throw std::invalid_argument("True labels and predicted labels must have the same size.");
    }

    size_t correct = 0;
    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (true_labels[i] == predicted_labels[i]) {
            correct++;
        }
    }

    return static_cast<double>(correct) / true_labels.size();
}

int main() {
    DataLoader loader;
    std::vector<std::vector<double>> data;
    std::vector<int> labels;

    // Load the Iris dataset
    loader.load_iris_dataset("C:/Users/User/CPU-GPU-DecisionTreeAlgorithm/iris.data", data, labels);

    // Print dataset size for debugging
    std::cout << "Loaded " << data.size() << " samples with " << data[0].size() << " features each." << std::endl;

    // Split the dataset into training and testing sets
    std::vector<std::vector<double>> train_data, test_data;
    std::vector<int> train_labels, test_labels;
    split_dataset(data, labels, train_data, train_labels, test_data, test_labels);

    std::cout << "Training set size: " << train_data.size() << std::endl;
    std::cout << "Testing set size: " << test_data.size() << std::endl;

    // Train the Decision Tree
    DecisionTree tree;
    tree.fit(train_data, train_labels);

    // Test the Decision Tree on the testing set
    std::vector<int> predicted_labels;
    for (const auto& sample : test_data) {
        int prediction = tree.predict(sample);
        predicted_labels.push_back(prediction);
        std::cout << "Predicted: " << prediction << ", True: " << test_labels[predicted_labels.size() - 1] << std::endl;
    }

    // Calculate accuracy
    double accuracy = calculate_accuracy(test_labels, predicted_labels);
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    return 0;
}