#-------------------------------------------------------------------------
# AUTHOR: Cyenadi Greene
# FILENAME: naive_bayes.py
# SPECIFICATION: reads training and test data from csv files, trains a Gaussian Naive Bayes classifier, and makes predictions with confidence scores
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
outlook_mapping = {"Sunny": 1, "Overcast": 2, "Rain": 3}
tempurature_mapping = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity_mapping = {"High": 1, "Normal": 2}
wind_mapping = {"Weak": 1, "Strong": 2}
class_mapping = {"Yes": 1, "No": 2}
reverse_class_mapping = {1: "Yes", 2: "No"}

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here

X =[]
Y=[]
for row in dbTraining:
    X.append([
        outlook_mapping[row[1]],
        tempurature_mapping[row[2]],
        humidity_mapping[row[3]],
        wind_mapping[row[4]]
    ])
    Y.append(class_mapping[row[5]])
#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
print("Day\tOutlook\tTemperature\tHumidity\tWind\tPlayTennis\tConfidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in dbTest:
    test_sample = [
        outlook_mapping[row[1]],
        tempurature_mapping[row[2]],
        humidity_mapping[row[3]],
        wind_mapping[row[4]]
    ]
    probs = clf.predict_proba([test_sample])[0]
    predicted_class= clf.predict([test_sample])[0]
    confidence = max(probs)

    if confidence >= 0.75:
        print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{reverse_class_mapping[predicted_class]}\t{confidence:.2f}")


