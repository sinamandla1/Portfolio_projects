import pandas as pd
import numpy as np

#Take off display limitations on Pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

oil = pd.read_csv('oil.csv')

#add a time feature for each of the prices recorded
oil['time'] = np.arange(len(oil.index))
oil.head()

#Perform EDA to see how many values are missing
oil.info()

#drop thos specific NA values
new_oil = oil.dropna(subset=['dcoilwtico'])
new_oil.info()

#visualize the data using a scatter plot
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plotting style and parameters using Seaborn
sns.set(style="whitegrid")
sns.set_context("notebook", rc={"figure.autolayout": True, "figure.figsize": (11, 4),
                                "axes.labelweight": "bold", "axes.labelsize": "large",
                                "axes.titleweight": "bold", "axes.titlesize": 16, "axes.titlepad": 10})

# Create a figure and axis
fig, ax = plt.subplots()

sns.regplot(x='time', y='dcoilwtico', data=new_oil, ci=None, scatter_kws=dict(color='0.25'), ax=ax)

# Set the title
ax.set_title('Time Plot of Oil Prices')

# Show the plot
plt.show()

#Notice we have a downward trend for the prices and they consolidate between $60-$20
#now move on to create a lag shift feature with a 1 day shift
oil['lag_1'] = oil['dcoilwtico'].shift(1)
lag_oil = oil.reindex(columns=['dcoilwtico', 'lag_1'])

#observe the values and see the correlation
lag_oil.describe()

#drop all NA values and then store the new df in a variable and continue
new_lag_oil = oil.dropna(subset=['dcoilwtico', 'lag_1'])

#we can see a direct correlation in the values meaning that the oil prices
##do depend on the previous days closing price based on the visualization we
## have a strong upward trend
fig, ax = plt.subplots()
ax = sns.regplot(x='lag_1', y='dcoilwtico', data=new_lag_oil, ci=None, scatter_kws=dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag Plot of Oil prices')

#Oil conclusion: the lag feature is directly correlated with the original
##oil prices do indeed depend heavily on the previous time step.
##We can also see that the oil prices do not move too much and we can use either Time 
##or Lag shift to train a model to predict values.
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

# Extracting features (X) and target (y)
X = new_lag_oil[['time']]
y = new_lag_oil['dcoilwtico']

# Aligning X and y based on their common index
y, X = y.align(X, join='inner')

# Creating an XGBoost regressor
model_oil = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

# Fitting the model
model_oil.fit(X, y)

# Making predictions
y_pred = pd.Series(model_oil.predict(X), index=X.index)

# Creating a lag plot
fig, ax = plt.subplots()
ax.plot(X['time'], y, '.', color='0.25', label='Actual')
ax.plot(X['time'], y_pred, label='Predicted')
ax.set_aspect('equal')
ax.set_ylabel('dcoilwtico')
ax.set_xlabel('time')
ax.set_title('time Plot of oil prices')
ax.legend()
plt.show()

##Visualize the results of the model
import matplotlib.pyplot as plt

# Set your plot parameters as needed
plot_params = {
    'figsize': (10, 6),  # Figure size
    'title': 'Predicted Values',  # Title
    'xlabel': 'Index (or time)',  # X-axis label
    'ylabel': 'dcoilwtico',  # Y-axis label
}

moving_average = y_pred.rolling(
    window=242,       # 242-day window
    center=True,      # puts the average at the center of the window
    min_periods=121,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

# Create a plot
ax = y.plot(**plot_params, label='Actual')  # Plot actual values
y_pred.plot(ax=ax, label='Predicted')  # Plot predicted values on the same axis
moving_average.plot(ax=ax, label='Moving Average')  # Plot the moving average on the same axis


# Add a legend
ax.legend()

# Display the plot
plt.show()

#As we can see the model could predict the values very accurately
#DONE
