# Decision Tree Classifier Algorithm with OpenMP Parallelization

## Overview
This is a project created for the course of High Performance Computer Architecture 2024/2025 that implements a Decision Tree Classifier using C++ with OpenMP for parallelization. The classifier is trained on the Iris dataset, which consists of 4 numerical features and a class label-

## Project's features
- Loads and processes the Iris dataset from a CSV file
- Implements a Decision Tree algorithm for classification
- Uses OpenMP parallelization to speed up the tree-building process
- Splits the dataset into training and testing sets
- Computes model accuracy by comparing predicted vs. actual labels

## Project's structure
```
│── data_loader.h       # Header file for dataset loading
│── data_loader.cpp     # Implementation file for dataset loading
│── decision_tree.h     # Header file for Decision Tree Classifier
│── decision_tree.cpp   # Implementation file for Decision Tree Classifier
│── decision_tree_without_openmp.cpp   # Implementation file for Decision Tree Classifier without OpenMP parallelization
│── iris.data           # Iris Dataset file
│── main.cpp            # Main program file
│── README.md           # Project documentation
```

## Prerequisites to run the project
- C++ Compiler (G++) with OpenMP support (e.g., GCC 9+ or Clang)

## Compilation & Execution
Using **GCC**:
```sh
g++ -o decision_tree -fopenmp main.cpp decision_tree.cpp data_loader.cpp
```

### Run the Program
```sh
./decision_tree
```

## Dataset
The code loads the Iris dataset from a CSV file, with each row containing:
```
SepalLength, SepalWidth, PetalLength, PetalWidth, Label
```
Example:
```
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
```

## OpenMP Parallelization
- The tree-building process is parallelized using OpenMP.
- The best feature and threshold selection is performed concurrently
- The following section in `decision_tree.cpp` handles parallelization:
```cpp
#pragma omp parallel
{
    #pragma omp for nowait
    for (int feature_index = 0; feature_index < num_features; feature_index++) {
        // Code for feature selection and Gini calculation
    }

    #pragma omp critical
    {
        // Updating global best split parameters safely
    }
}
```

## Performance Improvement
We noted that using OpenMP, the decision tree training process significantly reduced execution time. The parallelized feature selection loop speeds up the computation of the best split at each node.

## Authors
- Klejda Rrapaj: k.rrapaj@student.unisi.it
- Sildi Ricku: s.ricku@student.unisi.it
- Yelyzaveta Kasapien: y.kasapien@student.it