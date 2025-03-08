#include <iostream>
#include <vector>
#include <random>
#include "data_loader.h"
#include "decision_tree.h"
#include <algorithm>

/**
 * Function that splits the iris dataset into training and testing sets
 * @param data: The dataset (features)
 * @param labels: Corresponding class labels
 * @param train_data: Output vector for training features
 * @param train_labels: Output vector for training labels
 * @param test_data: Output vector for testing features
 * @param test_labels: Output vector for testing labels
 * @param train_ratio: The percentage of data used for training (default = 80%)
 */

void split_dataset(const std::vector<std::vector<double>>& data, const std::vector<int>& labels,
                   std::vector<std::vector<double>>& train_data, std::vector<int>& train_labels,
                   std::vector<std::vector<double>>& test_data, std::vector<int>& test_labels,
                   double train_ratio = 0.8) {
    std::random_device rd; // Random seed for shuffling
    std::mt19937 g(rd()); // Random number generator
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffling of the indices randomly
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

/**
 * Function that calculates the accuracy of predictions
 * @param true_labels: The actual class labels
 * @param predicted_labels: The predicted class labels
 * @return The accuracy as a percentage
 */
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

    // Loading the Iris dataset
    std::cout << "Loading dataset..." << std::endl;
    loader.load_iris_dataset("C:/Users/User/CPU-GPU-DecisionTreeAlgorithm/iris.data", data, labels);

    // Printing dataset size for debugging
    std::cout << "Loaded " << data.size() << " samples with " << data[0].size() << " features each." << std::endl;

    // Splitting the dataset into training and testing sets
    std::cout << "Splitting dataset into training and testing sets..." << std::endl;
    std::vector<std::vector<double>> train_data, test_data;
    std::vector<int> train_labels, test_labels;
    split_dataset(data, labels, train_data, train_labels, test_data, test_labels);

    std::cout << "Training set size: " << train_data.size() << std::endl;
    std::cout << "Testing set size: " << test_data.size() << std::endl;

    // Training the Decision Tree algorithm
    std::cout << "Training the decision tree..." << std::endl;
    DecisionTree tree;
    tree.fit(train_data, train_labels);
    std::cout << "Decision tree training completed." << std::endl;

    // Testting the Decision Tree on the testing set
    std::cout << "Predicting labels for the test set..." << std::endl;
    std::vector<int> predicted_labels;
    for (const auto& sample : test_data) {
        int prediction = tree.predict(sample);
        predicted_labels.push_back(prediction);
        std::cout << "Predicted: " << prediction << ", True: " << test_labels[predicted_labels.size() - 1] << std::endl;
    }

    // Calculating accuracy
    double accuracy = calculate_accuracy(test_labels, predicted_labels);
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    return 0;
}