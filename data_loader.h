#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>

class DataLoader {
public:
    void load_iris_dataset(const std::string& filepath, std::vector<std::vector<double>>& data, std::vector<int>& labels);
};

#endif