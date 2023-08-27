#Load the dataset 
import pandas as pd
import numpy as np

dataset = pd.read_csv("train.csv")
test_dataset = pd.read_csv("test.csv")

#Take off display limitations on Pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#Display the datasets and conduct Exploratory Data analysis
dataset.head()
test_dataset.head()

#Store the features into variables based on the D-type
#Notice that I excluded the features that will not be needed for modelling
##Initially I did not know that I should drop those features because I forgot
###That I could do a X.drop(["Cabin","VIP","Name"], axis=1, errors="ignore")
####to bypass the error that occurs
numerical_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
categorical_features = ["HomePlanet", "CryoSleep", "Destination"]

#We begine looking for null values in both variables and then replace them
##using specific strategies for each, Numerical needed a 'Mean' strategy
##Categorical features I used 'Mode' strategy. I built the necessary classes
##to be run in the pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class NumericalImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X[numerical_features]= imputer.fit_transform(X[numerical_features])
        return X

class categoricalImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        imputer = SimpleImputer(strategy="most_frequent")
        X[categorical_features]= imputer.fit_transform(X[categorical_features])
        return X

class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(["Cabin","VIP","Name"], axis=1, errors="ignore")

#create the pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([("NumericalImputer", NumericalImputer()),
                     ("categoricalImputer", categoricalImputer()),
                    ("FeatureDropper", FeatureDropper())])

#Run the test and train datasets throught th pipeline
processed_train_data = pipeline.fit_transform(dataset)
processed_test_data = pipeline.fit_transform(test_dataset)

#check the datasets
processed_train_data.info()
processed_test_data.info()

#During the EDA I noticed that the 'PassengerId' feature had a delimiter
#Therefore the procedure was to split it at the delim, and then create
#a split feature and store the values, and then convert the values for modeling
processed_train_data[['PassengerId_Part1', 'PassengerId_Part2']] = processed_train_data['PassengerId'].str.split('_', expand=True)
processed_test_data[['PassengerId_Part1', 'PassengerId_Part2']] = processed_test_data['PassengerId'].str.split('_', expand=True)

#Convert
processed_train_data['PassengerId_Part1'] = pd.to_numeric(processed_train_data['PassengerId_Part1'])
processed_train_data['PassengerId_Part2'] = pd.to_numeric(processed_train_data['PassengerId_Part2'])

processed_test_data['PassengerId_Part1'] = pd.to_numeric(processed_train_data['PassengerId_Part1'])
processed_test_data['PassengerId_Part2'] = pd.to_numeric(processed_train_data['PassengerId_Part2'])

#drop the old 'passengerid' feature
to_encode_train_data = processed_train_data.drop(["PassengerId"], axis=1, errors="ignore")
to_encode_test_data = processed_test_data.drop(["PassengerId"], axis=1, errors="ignore")

to_encode_train_data.head()

#I chose to create another pipeline since I was slowly grasping the concept of
#imputation. After doing Queries on the datasets on Excell/SQL I could see how
#many types of values were in the features that needed to be noted.
from sklearn.preprocessing import OneHotEncoder

class FeatureEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['CryoSleep']]).toarray()
        
        column_names = ['True', 'False']
        
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
            
        matrix = encoder.fit_transform(X[['HomePlanet']]).toarray()
        
        column_names = ['Earth', 'Europa', 'Mars']
        
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]

        matrix = encoder.fit_transform(X[['Destination']]).toarray()
        
        column_names = ['TRAPPIST-1e', '55 Cancri e', 'PSO J318.5-22']
        
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        return X

#created another feature dropper for the old columns
class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(['CryoSleep', 'HomePlanet', 'Destination'], axis=1, errors="ignore")

#created a second pipeline
encoder_pipeline = Pipeline([("featureEncoder", FeatureEncoder()),
                            ("FeatureDropper", FeatureDropper())])

#Run the preprocessed data throught the new pipeline
final_train_set = encoder_pipeline.fit_transform(to_encode_train_data)
final_test_set = encoder_pipeline.fit_transform(to_encode_test_data)

#check the data
final_test_set.info()
final_train_set.info()

#Time to model according to the processed train dataset:
##Begin by dropping the feature to be predicted
time_to_train = final_train_set.drop(['Transported'], axis=1, errors="ignore")

#Train the model using XGboost classifier since this is a classification
#problem
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#split the features
X = time_to_train
y = final_train_set['Transported']

#split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#make the XGboost model
xg_model = xgb.XGBClassifier()

#Train the model using the training data
xg_model.fit(X_train, y_train)

y_val_prediction = xg_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_prediction)

print("Validation Accuracy:", accuracy)
#the test produced: "Validation Accuracy: 0.780333525014376"

#All is good now to use the test dataset and notice, we had split the
#'Passengerid' feature so I must reverse the process with the new results
X_test = final_test_set

predictions = xg_model.predict(X_test)
predictions = predictions.astype(str)
predictions = np.char.replace(predictions, '0', 'False')
predictions = np.char.replace(predictions, '1', 'True')

#now create a data frame according to the requirements
predictions_df = pd.DataFrame()
predictions_df['PassengerId'] = test_dataset['PassengerId']
predictions_df['Transported'] = predictions

#check the result before production
predictions_df.head()

#Put the model into production and submit
predictions_df.to_csv('predictions.csv', index=False)
#DONE!
