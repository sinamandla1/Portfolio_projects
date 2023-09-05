import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

#Take off display limitations on Pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#supplementary data
holidays = pd.read_csv('holidays_events.csv')
stores = pd.read_csv('stores.csv')
oil = pd.read_csv('oil.csv')
transactions = pd.read_csv('transactions.csv')

#Convert the dat feature d-type to 'datetime' this is important to use in TS forecasting
train_df['date']=pd.to_datetime(train_df['date'])
test_df['date']=pd.to_datetime(test_df['date'])
holidays['date']=pd.to_datetime(holidays['date'])
transactions['date']=pd.to_datetime(transactions['date'])
oil['date']=pd.to_datetime(oil['date'])

#set the date as an index and this will help with EDA when we plot
oil.set_index('date', inplace=True)

# Create a new DataFrame with complete date range
date_range = pd.date_range(start=oil.index.min(), end=oil.index.max(), freq='D')
complete_oil = pd.DataFrame(index=date_range)

# Merge the original oil data with the complete date range DataFrame
complete_oil = complete_oil.merge(oil, left_index=True, right_index=True, how='left')

# Forward fill missing values to carry forward the last known value
complete_oil['dcoilwtico'].fillna(method='ffill', inplace=True)
complete_oil['dcoilwtico'].fillna(method='bfill', inplace=True)#fill the first date

# Reset index
complete_oil.reset_index(inplace=True)
complete_oil.rename(columns={'index': 'date'}, inplace=True)

#Get feature names and store them for future ease of use
train_columns = train_df.columns
test_columns = test_df.columns
stores_columns = stores.columns
oil_columns = oil.columns
holidays_columns = holidays.columns
transactions_columns = transactions.columns

#plot the links between each set and use this information to merge them

tables = {
    'train_df': train_columns,
     'test_df': test_df.columns,  # Add other fields
    'stores': stores.columns,  # Add other fields
    'oil': oil.columns,  # Add other fields
    'holidays': holidays.columns,  # Add other fields
    'transactions':transactions.columns
}
# Create a directed graph
G = nx.DiGraph()

# Add nodes (tables)
for table in tables:
    G.add_node(table)

# Add edges (common fields)
for table1 in tables:
    for table2 in tables:
        if table1 != table2:
            common_fields = set(tables[table1]) & set(tables[table2])
            if common_fields:
                G.add_edge(table1, table2, common_fields=common_fields)

# Plot the graph
pos = nx.spring_layout(G)  # Layout for better visualization
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, node_color='lightblue')
edge_labels = nx.get_edge_attributes(G, 'common_fields')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
plt.title("Common Fields between Tables")
plt.show()

#merge the datasets into one big one for Train and Test
frames = [train_df, complete_oil]

train = pd.merge(train_df, stores, on='store_nbr', how='left')
train = pd.concat(frames, axis=1)
train = pd.merge(train_df, holidays, on='date', how='left')

test = pd.merge(test_df, stores, on='store_nbr', how='left')
test = pd.concat(frames, axis=1)
test = pd.merge(test_df, holidays, on='date', how='left')

#Since .info() doesnt show enough data on the missing data we need to do it
#differently
# Count the number of missing values in each column

train_missing_values = train.isnull().sum()
test_missing_values = test.isnull().sum()

print("Missing Values in Train DataFrame:")
print(train_missing_values)
print("\nMissing Values in Test DataFrame:")
print(test_missing_values)

from sklearn.preprocessing import LabelEncoder
#Select the 'Object' features and then do feature encoding
non_numeric_columns = train.select_dtypes(exclude=['number']).columns.tolist()
non_numeric_columns=non_numeric_columns[1:]

label_encoder = LabelEncoder()
# Convert non-numeric columns to numeric labels
for col in non_numeric_columns:
    train[col] = label_encoder.fit_transform(train[col])
    test[col] = label_encoder.transform(test[col])

#check if the job is done
train_missing_values = train.isnull().sum()
test_missing_values = test.isnull().sum()

print("Missing Values in Train DataFrame:")
print(train_missing_values)
print("\nMissing Values in Test DataFrame:")
print(test_missing_values)

#Since all the data has been cleaned and converted for modelling
#we should look at correlation and decide which features are relevant
train_to_see_corr = train.drop(['date'], axis=1, errors="ignore")
correlation_matrix = train_to_see_corr.corr()
correlation_with_target = correlation_matrix['sales'].sort_values(ascending=False)
# Plot heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

#split the date into specific features from that date

def create_features(data):
    data=data.copy()
    data['date']=pd.to_datetime(data['date'])                                                  
    data['dayofweek']=data['date'].dt.dayofweek
    data['quarter']=data['date'].dt.quarter
    data['day']=data['date'].dt.day
    data['month']=data['date'].dt.month
    data['year']=data['date'].dt.year
        
    
    return data

#run the train and test set through the function
train=create_features(train)
test=create_features(test)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

#define a score function for each training split
def rmsle_score(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle_score, greater_is_better=False)

model_features = ['id', 'store_nbr', 'family', 'onpromotion', 'type', 'locale', 'locale_name',
                  'description', 'transferred', 'dayofweek', 'quarter', 'day', 'month', 'year']

# get different "family" value
family_values = train["family"].unique()

# initialize dict for each model and score
family_models = {}
family_scores={}

#train for each "family" model
for family_value in family_values:
    family_data = train[train["family"] == family_value]
    
    # split
    X = family_data[model_features].drop(columns=['family'])  
    y = family_data["sales"]  
    
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # save model
    family_models[family_value] = model
    
     # predict the x_valid
    y_pred = model.predict(X_valid)
    y_pred = np.maximum(y_pred, 0)#less than 0 become 0
    
    # calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y_valid, y_pred))
    family_scores[family_value] = rmsle

    # output each "family"  RMSLE score
    print(f"Family: {family_value}, RMSLE: {rmsle}")
  
average_rmsle = np.mean(list(family_scores.values()))
print(f"Average RMSLE for all families: {average_rmsle}")

# initialize an empty dictionary to store the avg sales values for
#different family values
family_mean_sales = {}

for family_value in family_values:
    family_data = train[train["family"] == family_value]
    mean_sales = family_data["sales"].mean()
    family_mean_sales[family_value] = mean_sales

#store the features that will be used
test_dataset = test[model_features]

#Create the Dataframe for the submission
predictions = pd.DataFrame()

#initialize an empty list to store the values
test_predictions = []

#get the DISTINCT products for the values to be predicted for each, in total there are 5
test_family_values = test['family'].unique()

#create a loop for the model to go through each product and do its prediction
#if there is a value then it will work with it, or else it will assign a default value based on the training split before
for test_family_value in test_family_values:
    test_family_data = test[test['family'] == test_family_value]
    X_test = test_family_data[model_features].drop(['family'], axis=1, errors="ignore")
    model = family_models.get(test_family_value)
    if model is not None:
        prediction = model.predict(X_test)
        prediction = np.maximum(prediction, 0)
        test_family_data = test_family_data.copy()
        test_family_data.loc[:, 'sales'] = prediction
        predictions = pd.concat([predictions, test_family_data], ignore_index=True)
        test_predictions.append(prediction)
    else:
        default_prediction = family_mean_sales.get(family_value, DEFAULT_PREDICTION)
        test_family_data.loc[:, 'sales'] = default_prediction
        predictions = pd.concat([predictions, test_family_data], ignore_index=True)
        test_predictions.append(default_prediction)

#structure the CSV file to be submitted
predictions = predictions[['id', 'sales']]
predictions = predictions.sort_values(by='id')

predictions.to_csv('predictions.csv', index=False)
#DONE!
