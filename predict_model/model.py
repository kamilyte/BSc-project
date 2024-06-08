import sys
sys.path.append('/Users/kamile/Desktop/Bachelor-Project/bachelor-project')
from data.database import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

def average_hindex(hindex_data):
    new_column = []
    for hindex_list in hindex_data:
        average = 0
        if hindex_list:
            sum = 0
            size = len(hindex_list)
            for h_index in hindex_list:
                sum += h_index
            average = sum / size
        new_column.append(average)
    return new_column
            

def max_hindex(hindex_data):
    new_column = []
    for hindex_list in hindex_data:
        max_h_index = 0
        if hindex_list:
            max_h_index = max(hindex_list)
        new_column.append(max_h_index)
    return new_column
    

data = {
    'current_citations': [10, 25, 5, 15, 30, 50, 3, 7, 20, 40],
    'altmetrics': [120, 250, 80, 150, 300, 500, 60, 110, 200, 400],
    'h_index': [5, 15, 3, 10, 20, 25, 2, 4, 18, 30],
    'future_citations': [20, 40, 10, 25, 60, 90, 8, 15, 45, 85]
}


df = pd.DataFrame(data)

x = df[['current_citations', 'altmetrics', 'h_index']]
y = df['future_citations']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predict the future number of citations for a new paper
# Example new paper with current metrics
new_paper = [[10, 50, 20]]  # current_citations, altmetrics, h_index
predicted_citations = model.predict(new_paper)
print(f'Predicted future citation count: {predicted_citations[0]}')




model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predict the future number of citations for a new paper
# Example new paper with current metrics
new_paper = [[10, 50, 20]]  # current_citations, altmetrics, h_index
predicted_citations = model.predict(new_paper)
print(f'Predicted future citation count: {predicted_citations[0]}')