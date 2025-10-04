#-------------------------------------------------------------------------
# AUTHOR: Cyenadi Greene
# FILENAME: knn.py
# SPECIFICATION: K-Nearest Neighbors (KNN) classifier for email classification using leave-one-out cross-validation
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
db = df.values.tolist()



num_errors = 0
num_samples = len(db)



#Loop your data to allow each instance to be your test set
for i in range(num_samples):

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    X_train=[ [float(val) for val in row[:-1]] for j, row in enumerate(db) if j != i ]
    Y_train=[ row[-1] for j, row in enumerate(db) if j != i ]


    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    test_sample = [float(val) for val in db[i][:-1]]
    true_label = db[i][-1]

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here

    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(X_train, Y_train)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here

    class_predicted = clf.predict([test_sample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
   
    if class_predicted != true_label:
        num_errors += 1

   
    #Print the error rate
    #--> add your Python code here 
error_rate = num_errors / num_samples
print(f"Error rate: {error_rate:.2f}")
































