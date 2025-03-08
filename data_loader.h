#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>

/**
 * This class provides a method to load the Iris dataset from a CSV file.
 */

class DataLoader {
public:
    void load_iris_dataset(const std::string& filepath, std::vector<std::vector<double>>& data, std::vector<int>& labels);
};

#endif