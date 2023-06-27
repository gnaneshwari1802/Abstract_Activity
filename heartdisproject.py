
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:29:05 2019

@author: M GNANESHWARI
"""
"""
Dataset :
The dataset consists of 779 individuals data. There are 15 columns in the dataset, however the first 
column name is not a good parameter as far as machine learning is considered so, there are effectively 14 columns.
Bussines Problem
Machine Learning End to End Project
Import essential libraries
Import dataset & explore
Know about dataset
Data visualization
Heatmap using the correlation matrix
Pair plot of fineTech_appData2
Countplot of enrolled
Histogram of each feature of fineTech_appData2
Correlation barplot with ‘enrolled’ feature
Feature selection
Heatmap with the correlation matrix
Data preprocessing
Split dataset in Train and Test
Feature Scaling
Machine Learning Model Building
Decision Tree Classifier
K – Nearest Neighbor Classifier
Naive Bayes Classifier
Random Forest Classifier
Logistic Regression
Support Vector Classifier
XGBoost Classifier
Confusion Matrix
Classification report of ML model
Cross-validation of the ML model
Mapping predicted output to the target
Save the Machine Learning model
Save the ML model with Pickle
Save the Ml model with Joblib
Conclusion
"""
"""
Age : displays the age of the individual.
Sex : displays the gender of the individual using the following format :
1 = male
0 = female
Chest-pain type : displays the type of chest-pain experienced by the individual using the following format :
1 = typical angina
2 = atypical angina
3 = non — anginal pain
4 = asymptotic
Resting Blood Pressure : displays the resting blood pressure value of an individual in mmHg (unit)
Serum Cholestrol : displays the serum cholestrol in mg/dl (unit)
Fasting Blood Sugar : compares the fasting blood sugar value of an individual with 120mg/dl.
If fasting blood sugar > 120mg/dl then : 1 (true)
else : 0 (false)
Resting ECG : displays resting electrocardiographic results
0 = normal
1 = having ST-T wave abnormality
2 = left ventricular hyperthrophy
Max heart rate achieved : displays the max heart rate achieved by an individual.
Exercise induced angina :
1 = yes
0 = no
ST depression induced by exercise relative to rest : displays the value which is integer or float.
Peak exercise ST segment :
1 = upsloping
2 = flat
3 = downsloping
Number of major vessels (0–3) colored by flourosopy : displays the value as integer or float.
Thal : displays the thalassemia :
3 = normal
6 = fixed defect
7 = reversible defect
Diagnosis of heart disease : Displays whether the individual is suffering from heart disease or not :
0 = absence
1, 2, 3, 4 = present.
"""
"""
Why these parameters:
In the actual dataset we had 76 features but for our study we chose only the above 14 because :
Age : Age is the most important risk factor in developing cardiovascular or heart diseases, with approximately a tripling of risk with each decade of life. Coronary fatty streaks can begin to form in adolescence. It is estimated that 82 percent of people who die of coronary heart disease are 65 and older. Simultaneously, the risk of stroke doubles every decade after age 55.
Sex : Men are at greater risk of heart disease than pre-menopausal women. Once past menopause, it has been argued that a woman’s risk is similar to a man’s although more recent data from the WHO and UN disputes this. If a female has diabetes, she is more likely to develop heart disease than a male with diabetes.
Angina (Chest Pain) : Angina is chest pain or discomfort caused when your heart muscle doesn’t get enough oxygen-rich blood. It may feel like pressure or squeezing in your chest. The discomfort also can occur in your shoulders, arms, neck, jaw, or back. Angina pain may even feel like indigestion.
Resting Blood Pressure : Over time, high blood pressure can damage arteries that feed your heart. High blood pressure that occurs with other conditions, such as obesity, high cholesterol or diabetes, increases your risk even more.
Serum Cholestrol : A high level of low-density lipoprotein (LDL) cholesterol (the “bad” cholesterol) is most likely to narrow arteries. A high level of triglycerides, a type of blood fat related to your diet, also ups your risk of heart attack. However, a high level of high-density lipoprotein (HDL) cholesterol (the “good” cholesterol) lowers your risk of heart attack.
Fasting Blood Sugar : Not producing enough of a hormone secreted by your pancreas (insulin) or not responding to insulin properly causes your body’s blood sugar levels to rise, increasing your risk of heart attack.
Resting ECG : For people at low risk of cardiovascular disease, the USPSTF concludes with moderate certainty that the potential harms of screening with resting or exercise ECG equal or exceed the potential benefits. For people at intermediate to high risk, current evidence is insufficient to assess the balance of benefits and harms of screening.
Max heart rate achieved : The increase in the cardiovascular risk, associated with the acceleration of heart rate, was comparable to the increase in risk observed with high blood pressure. It has been shown that an increase in heart rate by 10 beats per minute was associated with an increase in the risk of cardiac death by at least 20%, and this increase in the risk is similar to the one observed with an increase in systolic blood pressure by 10 mm Hg.
Exercise induced angina : The pain or discomfort associated with angina usually feels tight, gripping or squeezing, and can vary from mild to severe. Angina is usually felt in the centre of your chest, but may spread to either or both of your shoulders, or your back, neck, jaw or arm. It can even be felt in your hands. o Types of Angina a. Stable Angina / Angina Pectoris b. Unstable Angina c. Variant (Prinzmetal) Angina d. Microvascular Angina.
Peak exercise ST segment : A treadmill ECG stress test is considered abnormal when there is a horizontal or down-sloping ST-segment depression ≥ 1 mm at 60–80 ms after the J point. Exercise ECGs with up-sloping ST-segment depressions are typically reported as an ‘equivocal’ test. In general, the occurrence of horizontal or down-sloping ST-segment depression at a lower workload (calculated in METs) or heart rate indicates a worse prognosis and higher likelihood of multi-vessel disease. The duration of ST-segment depression is also important, as prolonged recovery after peak stress is consistent with a positive treadmill ECG stress test. Another finding that is highly indicative of significant CAD is the occurrence of ST-segment elevation > 1 mm (often suggesting transmural ischaemia); these patients are frequently referred urgently for coronary angiography.
The Approach
The the code for this article can be found here.
The code is implemented in Python and different classification models are applied.
In this article I will be using the following classification models for classification :
SVM
Naive Bayes
Logistic Regression
Decision Tree
Random Forest
LightGBM
XGboost
Data Analysis
Let us look at the people’s age who are suffering from the disease or not.
Here, target = 1 implies that the person is suffering from heart disease and target = 0 implies the person is not suffering.

"""
"""
Model Training and Prediction :
We can train our prediction model by analyzing existing data because we already know whether each patient has heart disease. This process is also known as supervision and learning. The trained model is then used to predict if users suffer from heart disease. The training and prediction process is described as follows:

Splitting:
First, data is divided into two parts using component splitting. In this experiment, data is split based on a ratio of 80:20 for the training set and the prediction set. The training set data is used in the logistic regression component for model training, while the prediction set data is used in the prediction component.

The following classification models are used - Logistic Regression, Random Forest Classfier, SVM, Naive Bayes Classifier, Decision Tree Classifier, LightGBM, XGBoost

Prediction:
The two inputs of the prediction component are the model and the prediction set. The prediction result shows the predicted data, actual data, and the probability of different results in each group.

Evaluation:
The confusion matrix, also known as the error matrix, is used to evaluate the accuracy of the model.


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as p
import seaborn as sns
#import pyreadstat
from dateutil import parser # convert time in date time data type
data_cars=pd.read_csv("D:\\DataScience\\heart-disease-dataset\\heart.csv")
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)
"""
print(data_cars.dtypes)

print(data_cars)
print(data_cars.info())
print(data_cars.describe())
# sns heatmap correlation
 
p.figure(figsize=(16,9))
 
sns.heatmap(data_cars.corr(), annot = True)
p.figure(figsize=(16,9))
 
corr_mx = data_cars.corr() # correlation matrix
 
matrix = np.tril(corr_mx) # take lower correlation matrix
 
sns.heatmap(corr_mx, mask=matrix)










# to get 3 random rows
# each time you run this, you would have 3 different rows
print(data_cars.head(7))
print(data_cars.tail())
### 1 = male, 0 = female
print(data_cars.isnull().sum())
"""
#data_cars.hist()
"""print(data_cars.head())
print(data_cars.size)
print(data_cars.columns)
"""
print(data_cars.info())
print(data_cars.describe())

print(data_cars.describe().transpose())
pd.set_option('display.float_format',lambda x:'%.3f'%x)
data_cars=data_cars.drop(['ca'],axis=1)
print(data_cars.info())
print(data_cars.size)
print(data_cars.shape)
################################## data preprocessing
"""X = data_cars.iloc[:, :-1].values
y = data_cars.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

"""
data_cars.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']




data_cars.['target'] = data_cars.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
data_cars.['sex'] = data_cars.sex.map({0: 'female', 1: 'male'})
data_cars.['thal'] = data_cars.thal.fillna(data_cars.thal.mean())
data_cars.['ca'] = data_cars.ca.fillna(data_cars.ca.mean())


# distribution of target vs age 
sns.set_context("paper", font_scale = 2, rc = {"font.size": 20,"axes.titlesize": 25,"axes.labelsize": 20}) 
sns.catplot(kind = 'count', data = data_cars., x = 'age', hue = 'target', order = data_cars.['age'].sort_values().unique())
plt.title('Variation of Age for each target class')
plt.show()

 
# barplot of age vs sex with hue = target
sns.catplot(kind = 'bar', data = data_cars., y = 'age', x = 'sex', hue = 'target')
plt.title('Distribution of age vs sex with the target class')
plt.show()

data_cars.['sex'] = data_cars.sex.map({'female': 0, 'male': 1})




#########################################   SVM   #############################################################
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))


#########################################   Naive Bayes  #############################################################
X = data_cars.iloc[:, :-1].values
y = data_cars.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Naive Bayes = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Naive Bayes = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))


#########################################   Logistic Regression  #############################################################
X = data_cars.iloc[:, :-1].values
y = data_cars.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

#########################################   Decision Tree  #############################################################
X = data_cars.iloc[:, :-1].values
y = data_cars.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Decision Tree = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))


#########################################  Random Forest  #############################################################
X = data_cars.iloc[:, :-1].values
y = data_cars.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Random Forest = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Random Forest = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

###############################################################################
# applying lightGBM
import lightgbm as lgb

d_train = lgb.Dataset(X_train, label = y_train)
params = {}

clf = lgb.train(params, d_train, 100)
#Prediction
y_pred = clf.predict(X_test)
#convert into binary values
for i in range(0, len(y_pred)):
    if y_pred[i]>= 0.5:       # setting threshold to .5
       y_pred[i]=1
    else:  
       y_pred[i]=0
       
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = clf.predict(X_train)

for i in range(0, len(y_pred_train)):
    if y_pred_train[i]>= 0.5:       # setting threshold to .5
       y_pred_train[i]=1
    else:  
       y_pred_train[i]=0
       
cm_train = confusion_matrix(y_pred_train, y_train)
print()
print('Accuracy for training set for LightGBM = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for LightGBM = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))


###############################################################################
# applying XGBoost

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.20, random_state = 0)


from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = xg.predict(X_train)

for i in range(0, len(y_pred_train)):
    if y_pred_train[i]>= 0.3:       # setting threshold to .5
       y_pred_train[i]=1
    else:  
       y_pred_train[i]=0
       
cm_train = confusion_matrix(y_pred_train, y_train)
print()
print('Accuracy for training set for XGBoost = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for XGBoost = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

"""