import sys
sys.path.append('/Users/kamile/Desktop/Bachelor-Project/BSc-project')
from data_visualisation.data_analysis import normalised_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_log_error
#from sklearn.metrics import huber_loss

def get_relevant_columns():
    df = normalised_data()
    #df = df.drop(df[df['impact'] == 'low'].index)
    #df = df.drop(["doi", "title", "year", "query", "total_citations", "limited_citations", "impact", "age", "first_year_citations", "second_year_citations", "text"], axis=1)
    df = df.drop(["doi", "title", "year", "query", "total_citations", "limited_citations", "impact", "age", "normalised_first_year", "normalised_second_year", "text"], axis=1)
    return df

def huber_loss(y_pred, y_true, delta=1.0):
    error = y_pred - y_true
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    return np.mean(0.5 * quadratic ** 2 + delta * linear)
    
def perform_regression():
    df = get_relevant_columns()
    #print(df.head())
    #X = df[['avg_h_index', 'usage', 'captures', 'mentions', 'social_media', 'max_h_index', 'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog_index', 'smog_index', 'automated_readability_index', 'coleman_liau_index', 'cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions', 'normalised_first_year', 'normalised_second_year']]
    #X = df[['normalised_first_year', 'normalised_second_year', 'avg_h_index', 'max_h_index']]
    X = df[['first_year_citations', 'second_year_citations', 'avg_h_index', 'max_h_index']]
    y = df[["normalised_citations"]]
    # df_train, df_temp, y_train, y_temp = train_test_split(X,
    #                                                       y,
    #                                                       stratify=y,
    #                                                       test_size=(1.0 - 0.2),
    #                                                       random_state=42)
    
    # df_val, df_test, y_val, y_test = train_test_split(df_temp,
    #                                                   y_temp,
    #                                                   stratify=y_temp,
    #                                                   test_size=0.5,
    #                                                   random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    
    model = LinearRegression()
    #model = HuberRegressor()
    #model = RandomForestRegressor(n_estimators=1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    median_ae = median_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    msle = mean_squared_log_error(y_test, y_pred)
    huber = huber_loss(y_pred, y_test)
    
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R Squared Error: {r2}')
    print(f'Mean Absolute Percentage Error: {mape}')
    print(f'Median Absolute Error: {median_ae}')
    print(f'Explained Variance Score: {evs}')
    print(f'Mean Squared Logarithmic Error: {msle}')
    print(f'Huber Loss: {huber}')
    
    # scoring = "neg_root_mean_squared_error"
    # scores = cross_validate(model, X_train, y_train, scoring=scoring, return_estimator=True)
    # print(scores)
    
    new_paper = [[1, 5, 19, 32]]
    predicted_citations = model.predict(new_paper)
    print(f'Predicted future citation count: {predicted_citations[0]}')
    

    
    
perform_regression()