#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>

/**
 * This class loads the Iris dataset from a CSV file
 * Each row of the Iris dataset consists of 4 numerical features followed by a label
 * The function here stores feature values in `data` and class labels in `labels`
 *  
 */

void DataLoader::load_iris_dataset(const std::string& filepath, std::vector<std::vector<double>>& data, std::vector<int>& labels) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return;
    }

    std::string line;

    // Reading the file line by line
    while (std::getline(file, line)) {
        if (line.empty()) continue; 

        std::stringstream ss(line);
        std::string token;
        std::vector<double> row;

        try {
            // Reading the first 4 columns that will be the numerical features
            for (int i = 0; i < 4; ++i) {
                if (!std::getline(ss, token, ',')) {
                    throw std::invalid_argument("Invalid data format");
                }
                row.push_back(std::stod(token));
            }

            data.push_back(row);

            // Reading the last column that will be the class label
            if (!std::getline(ss, token, ',')) {
                throw std::invalid_argument("Invalid label format");
            }

            // The process of class encoding is taking place here by converting string labels to an integer
            if (token == "Iris-setosa") labels.push_back(0);
            else if (token == "Iris-versicolor") labels.push_back(1);
            else if (token == "Iris-virginica") labels.push_back(2);
            else throw std::invalid_argument("Unknown label: " + token);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error parsing line: " << line << " - " << e.what() << std::endl;
        }
    }

    file.close();
}