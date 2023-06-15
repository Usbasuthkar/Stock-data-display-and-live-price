import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime
now = datetime.now();
date = now.day
month = now.month
year =  now.year
start_date = "2000-01-01"
end_date = str(year)+"-"+str(month)+"-"+str(date)
company_name = "HUDCO.NS"
stock_data = yf.download(company_name,start_date,end_date)
#stock_data = np.genfromtxt('stock_data.csv', delimiter=',')

# Split the data into input and output variables
X = stock_data[:, :-1]
y = stock_data[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Use the model to make predictions on the test data
predictions = model.predict(X_test)

# Print the accuracy of the model
accuracy = model.score(X_test, y_test)
print('Accuracy: ', accuracy)
