# -*- coding: utf-8 -*-
"""Model3_DecisionTreeClassifier

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kN44LHgrXvL-TRtRx-tYzxldqz_EhCMG
"""

import pandas as pd
import numpy as np
from google.colab import files
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
from google.colab import drive

uploaded = files.upload()
df = pd.read_csv(next(iter(uploaded)))
df.head()

class_name = df['pose_name'].unique()
print(class_name)

#assigning the values of the df to a numpy array
data=df.values

num_rows = data.shape[0]
num_cols = data.shape[1]

print(num_rows)
print(num_cols)

#features= pose landmarks, target= pose name
features = data[:, 1:]
target = data[:, 0]

#splitting the data
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2,random_state=42, shuffle= True)

#Model #3 Decision Tree Classifier
model_DTC= DecisionTreeClassifier()
model_DTC.fit(features_train, target_train)

prediction_DTC= model_DTC.predict(features_test)
print(accuracy_score(target_test, prediction_DTC)*100)

#looping through incorrect predictions
for i in range(len(prediction_DTC)):
  if target_test[i] != prediction_DTC[i]:
    print(target_test[i],prediction_DTC[i])

#for analyzing results
probability_scores = model_DTC.predict_proba(features_test)
predictions_DTC = model_DTC.predict(features_test)

DTC_model_results = []
for i in range(len(predictions_DTC)):
    predicted_val = predictions_DTC[i]
    label_val = target_test[i]
    arr = probability_scores[i]
    formatted_percentage_output = [round(value * 100, 2) for value in arr]  #convert probabilities to percentages
    formatted_percentage_output.append(predicted_val)  #append predicted label
    formatted_percentage_output.append(label_val)  #append actual label
    if predicted_val == label_val:
        formatted_percentage_output.append('YES')  #correct prediction
    else:
        formatted_percentage_output.append('NO')  #incorrect prediction

    DTC_model_results.append(formatted_percentage_output)

#create a df from the Decision Tree model results
DTC_df = pd.DataFrame(DTC_model_results,
                      columns=['cat pose', 'chair pose', 'corpse pose', 'cow pose', 'downward dog pose', 'flat back pose', 'high lunge pose',
                               'knee to elbow plank pose', 'knees to chest pose', 'low lunge pose', 'mountain pose', 'runners lunge twist pose',
                               'seated spinal twist pose', 'side plank yoga pose', 'standing forward bend pose', 'tabletop pose', 'three legged dog pose',
                               'tip toe pose', 'tree pose', 'upper plank pose', 'predicted label', 'true label', 'accurate'])

#save to a CSV file
DTC_df.to_csv('DTC_model_results.csv', index=False)

#download the file (specific to Google Colab)
files.download('DTC_model_results.csv')

#save model
drive.mount('/content/drive')
dump(model_DTC, '/content/drive/My Drive//Model3_DecisionTree.joblib')