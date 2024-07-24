# -*- coding: utf-8 -*-
"""Model4_RandomForrestClassifier

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hysKlzZDUiRty5MGs1nM7NBKbC-psWwL
"""

import pandas as pd
import numpy as np
from google.colab import files
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

#splitting data
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2,random_state=42, shuffle= True)

#Model #4 Random Forest Classifier
model_RFC = RandomForestClassifier(n_estimators=100, random_state=35)
model_RFC.fit(features_train, target_train)

prediction_RFC= model_RFC.predict(features_test)
print(accuracy_score(target_test, prediction_RFC)*100)

#for analyzing the results
probability_scores = model_RFC.predict_proba(features_test)
RFC_model_results = []
for i in range(len(prediction_RFC)):
  predicted_val = prediction_RFC[i]
  label_val = target_test[i]
  arr = probability_scores[i]
  formatted_percentage_output = [round(value * 100, 2) for value in arr]
  formatted_percentage_output.append(predicted_val)
  formatted_percentage_output.append(label_val)
  if predicted_val == label_val:
    formatted_percentage_output.append('YES')
  else:
    formatted_percentage_output.append('NO')

  RFC_model_results.append(formatted_percentage_output)

#for analyzing the results
RFC_df = pd.DataFrame(RFC_model_results, columns= ['cat pose','chair pose','corpse pose','cow pose','downward dog pose','flat back pose','high lunge pose',\
                                                   'knee to elbow plank pose','knees to chest pose','low lunge pose','mountain pose','runners lunge twist pose',\
                                                   'seated spinal twist pose','side plank yoga pose','standing forward bend pose','tabletop pose','three legged dog pose',\
                                                   'tip toe pose','tree pose','upper plank pose','predicted label','true label','accurate'])
RFC_df.to_csv('RFC_model_results.csv', index=False)

files.download('/content/RFC_model_results.csv')

#looping through incorrect predictions
for i in range(len(prediction_RFC)):
  if target_test[i] != prediction_RFC[i]:
    print(target_test[i],prediction_RFC[i])

# saving model
drive.mount('/content/drive')
dump(model_RFC, '/content/drive/My Drive/model4_RFC.joblib')