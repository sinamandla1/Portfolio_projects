#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Use pandas to import file
dataset = pd.read_csv('train.csv')

#Check the data
dataset.head()

#Visualize the correlation within the dataset
numeric_columns = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
numeric_data = dataset[numeric_columns]
sns.heatmap(numeric_data.corr(), cmap="YlGnBu")
plt.show()

#Shuffle the data
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indices, test_indices in split.split(dataset, dataset[["Survived", "Pclass", "Sex"]]):
    strat_train_set = dataset.loc[train_indices]
    strat_test_set = dataset.loc[test_indices]

#Check if it worked to shuffle the table
strat_test_set

#Check to see how the distribution worked out
plt.subplot(1, 2, 1)
strat_train_set["Survived"].hist()
strat_train_set["Pclass"].hist()

plt.subplot(1, 2, 2)
strat_test_set["Survived"].hist()
strat_test_set["Pclass"].hist()
plt.show()

#use the info function to see if we have any missing data
strat_train_set

#In the next 2 cells we will create a data pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class AgeImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X["Age"] = imputer.fit_transform(X[['Age']])
        return X

#Ecode certain features numerically
from sklearn.preprocessing import OneHotEncoder

class FeatureEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['Embarked']]).toarray()
        
        column_names = ['C', 'S', 'Q', 'N']
        
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
            
        matrix = encoder.fit_transform(X[['Sex']]).toarray()
        
        column_names = ["Female", "Male"]
        
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        return X

class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(["Embarked", "Name", "Ticket", "Cabin", "Sex", "N"], axis=1, errors="ignore")

#Create a pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([("ageimputer", AgeImputer()),
                    ("featureencoder", FeatureEncoder()),
                    ("featuredropper", FeatureDropper())])

#Use the pipeline
strat_train_set = pipeline.fit_transform(strat_train_set)

#Once pipeline is complete then we check if the features are correct
strat_train_set

#Begin Model selection
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#split the features
X = strat_train_set.drop("Survived", axis=1)
y = strat_train_set["Survived"]

#split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#make the XGboost model
xg_model = xgb.XGBClassifier()

#Train the model using the training data
xg_model.fit(X_train, y_train)

#Evaluate the model on the validation data
y_val_prediction = xg_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_prediction)

print("Validation Accuracy:", accuracy)

#Check the test data
strat_test_set.info()

#process the test data through our pipeline
strat_test_set = pipeline.transform(strat_test_set)

#Check the data
strat_test_set.info()

#drop the survived label
X_test = strat_test_set.drop("Survived", axis=1)
y_test = strat_test_set["Survived"]

#Evaluate the model
test_accuracy = xg_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)

#ALL IS GOOD NOW USE THE ENTIRE SET provided test.csv
final_dataset = pd.read_csv('test.csv')
# Preprocess the entire dataset using the pipeline
processed_dataset = pipeline.transform(final_dataset)

#Run it through the model XGboost
predictions_final = xg_model.predict(processed_dataset)

#create a dataframe to see if its all good
predictions_df = pd.DataFrame()
predictions_df['PassengerId'] = processed_dataset['PassengerId']
predictions_df['Survived'] = predictions_final

#display results
predictions_df

#Create CSV file for submission
predictions_df.to_csv('predictions.csv', index=False)
#DONE!
