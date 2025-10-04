#-------------------------------------------------------------------------
# AUTHOR: Cyenadi Greene
# FILENAME: decision_tree_2.py
# SPECIFICATION: Decision tree classifier for contact lens data , chooses the average 
# accuracy as the final classification performance of each model
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3 hours
#-------------------------------------------------------------------------

from sklearn import tree
import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True) 

# List of training CSV files
dataSets = [
    'contact_lens_training_1.csv',
    'contact_lens_training_2.csv',
    'contact_lens_training_3.csv'
]

# Read the test CSV file
df_test = pd.read_csv('contact_lens_test.csv')
df_test.columns = df_test.columns.str.strip()

# Define mapping for all categorical columns
mappings = {
    'Age': {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3},
    'Spectacle Prescription': {"Myope": 1, "Hypermetrope": 2},
    'Astigmatism': {"Yes": 1, "No": 2},
    'Tear Production Rate': {"Normal": 1, "Reduced": 2},
    'Recommended Lenses': {"Yes": 1, "No": 2}
}

# Clean test set values
for col in df_test.columns:
    if df_test[col].dtype == object:
        df_test[col] = df_test[col].str.strip().str.capitalize()

# Apply mapping to test set
df_test_num = df_test.replace(mappings)
X_test = df_test_num[['Age', 'Spectacle Prescription', 'Astigmatism', 'Tear Production Rate']].values
Y_test = df_test_num['Recommended Lenses'].astype(int).values

# Loop through each training dataset
for ds in dataSets:
    # Read and clean training data
    df_train = pd.read_csv(ds)
    df_train.columns = df_train.columns.str.strip()
    for col in df_train.columns:
        if df_train[col].dtype == object:
            df_train[col] = df_train[col].str.strip().str.capitalize()
    
    # Apply mappings
    df_train_num = df_train.replace(mappings)
    X = df_train_num[['Age', 'Spectacle Prescription', 'Astigmatism', 'Tear Production Rate']].values
    Y = df_train_num['Recommended Lenses'].astype(int).values

    accuracies = []

    # Repeat training/testing 10 times
    for i in range(10):
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf.fit(X, Y)

        Y_pred = clf.predict(X_test)
        accuracy = np.mean(Y_pred == Y_test)
        accuracies.append(accuracy)

    final_accuracy = np.mean(accuracies)
    print(f"final accuracy when training on {ds}: {final_accuracy:.3f}")
