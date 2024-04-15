import streamlit as st
import pandas as pd
import numpy as np

class Node:
    def __init__(self, feature=None, value=None, results=None, true_branch=None, false_branch=None):
        self.feature = feature          # Index of feature to split on
        self.value = value              # Value of the feature to split on
        self.results = results          # Stores class label counts for leaf nodes
        self.true_branch = true_branch  # Subtree for instances where feature = value
        self.false_branch = false_branch  # Subtree for instances where feature != value

# Function to calculate entropy
def entropy(data):
    classes = np.unique(data[:, -1])
    entropy_val = 0
    for cls in classes:
        p = len(data[data[:, -1] == cls]) / len(data)
        entropy_val -= p * np.log2(p)
    return entropy_val

# Function to split data based on a feature and its value
def split_data(data, feature, value):
    true_data = data[data[:, feature] == value]
    false_data = data[data[:, feature] != value]
    return true_data, false_data

# Function to find the best split based on information gain
def find_best_split(data):
    best_gain = 0
    best_feature = None
    n_features = len(data[0]) - 1
    
    current_entropy = entropy(data)
    
    for feature in range(n_features):
        feature_values = np.unique(data[:, feature])
        
        for value in feature_values:
            true_data, false_data = split_data(data, feature, value)
            
            if len(true_data) == 0 or len(false_data) == 0:
                continue
            
            true_entropy = entropy(true_data)
            false_entropy = entropy(false_data)
            p = len(true_data) / len(data)
            info_gain = current_entropy - p * true_entropy - (1 - p) * false_entropy
            
            if info_gain > best_gain:
                best_gain = info_gain
                best_feature = (feature, value)
    
    return best_feature

# Function to build the decision tree
def build_tree(data):
    feature, value = find_best_split(data)
    
    if feature is None:
        classes, counts = np.unique(data[:, -1], return_counts=True)
        results = dict(zip(classes, counts))
        return Node(results=results)
    
    true_data, false_data = split_data(data, feature, value)
    
    true_branch = build_tree(true_data)
    false_branch = build_tree(false_data)
    
    return Node(feature, value, true_branch=true_branch, false_branch=false_branch)

# Function to predict class labels for a single instance
def predict(node, instance):
    if node.results is not None:
        return max(node.results, key=node.results.get)
    else:
        if instance[node.feature] == node.value:
            return predict(node.true_branch, instance)
        else:
            return predict(node.false_branch, instance)

# Function to predict class labels for a dataset
def decision_tree_predict(tree, test_data):
    predictions = []
    for instance in test_data:
        predictions.append(predict(tree, instance))
    return predictions

def main():
    st.title("ID3 Decision Tree Classifier")
    st.write("This app demonstrates the working of a Decision Tree Classifier using the ID3 algorithm.")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        # Read the data
        data = pd.read_csv(uploaded_file)

        # Display dataset
        st.subheader("Dataset")
        st.write(data)

        # Prepare data
        X = data.drop(columns=['target'])
        y = data['target']
        data = np.column_stack((X.values, y.values))  # Convert to numpy array

        # Build the decision tree
        tree = build_tree(data)

        # Example prediction
        example_instance = X.iloc[0].values
        prediction = predict(tree, example_instance)
        st.subheader("Example Prediction")
        st.write("Example instance:", example_instance)
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()

