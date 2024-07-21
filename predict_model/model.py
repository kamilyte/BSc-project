import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV, train_test_split
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
import psycopg2
from data.config import DB_NAME, DB_HOST, DB_USER, DB_PASS, DB_PORT

# fetch table values
def fetch_db(): 
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
  
        df = pd.read_sql_query("""SELECT * FROM scopus_database_v4;""",conn)
        
        conn.commit()
        
    except Exception as e:
        print("Error fetching data from train_database: ", e)
    finally:
        if conn:
            conn.close()
            print("Database connection closed")
            
    return df

def plotting(y_test, y_train, y_pred_test, y_pred_train):
    
    # plot residuals for training data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.residplot(x=y_pred_train, y=y_train - y_pred_train, lowess=True, line_kws={'color': 'red'}, color="lightseagreen")
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted (Training)')

    # plot residuals for testing data
    plt.subplot(1, 2, 2)
    sns.residplot(x=y_pred_test, y=y_test - y_pred_test, lowess=True, line_kws={'color': 'red'}, color="lightseagreen")
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted (Testing)')
    plt.tight_layout()
    plt.show()
    
    # plot prediction values against the actual values for test set
    plt.tight_layout()
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, color="lightseagreen")
    plt.plot(y_test, y_test, color='red', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True)
    plt.show()

# r-squared errors
def errors(y_test, y_pred_test, y_train, y_pred_train):
    print('r2_score_train:',r2_score(y_train, y_pred_train))
    print('r2_score_test:',r2_score(y_test, y_pred_test))

def extratrees():
    df = fetch_db()
    
    X = df[['max_hindex', 'avg_hindex', 'min_hindex', 'num_authors', 'flesch_reading_ease', 'combined_grade', 'cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions', 'compound_sentiment', 'query_label']]
    y = df[["third_year_total"]]
    subfields = df[['query']]
    y = y.values.ravel()
    
    # split into train and test sets 
    X_train, X_test, y_train, y_test, subfield_train, subfield_test = train_test_split(X, y, subfields, test_size=0.3, random_state=42, shuffle=True)

    # parameter grid search
    param_search = {
        "n_estimators": [50, 100, 200, 300, 400, 500],
        "max_depth": [None, 2, 5, 10],
        "min_samples_leaf": [2, 5, 10, 20],
        "min_samples_split": [2, 5, 10, 20, 40],
        "bootstrap": [False, True]
    }
    
    # chosen model
    model = ExtraTreesRegressor(random_state=42)
    #model = ExtraTreesRegressor(n_estimators=400, max_depth=5, min_samples_leaf=20, min_samples_split=20, bootstrap=True, random_state=42)
    
    # chosen cross validation method
    k_folds = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
    
    # set up grid search
    grid_search = GridSearchCV(estimator=model, cv=k_folds, param_grid=param_search, return_train_score=True, verbose=2)
    
    # fit grid search with training set
    grid_search.fit(X_train, y_train)
    
    # get the best parameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f'Best parameters: {best_params}')
    print(f'Best cross-validation R² score: {best_score}')

    # fit the model with the best parameters
    model = ExtraTreesRegressor(**grid_search.best_params_)
    
    # train the model
    model.fit(X_train, y_train)
    
    # get the models cross validation score
    scores = cross_val_score(model, X_train, y_train, cv = k_folds, scoring='r2')
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))
    print(f'Standard deviation of cross-validation scores: {scores.std()}')
    
    # predict for training and test sets
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # transform the values
    y_pred_train = np.array(y_pred_train)
    y_pred_test = np.array(y_pred_train)
    subfield_test = np.array(subfield_test)
    subfield_test = subfield_test.ravel()
    
    # get prediction accuracy for each subfield
    results = pd.DataFrame({'subfield': subfield_test, 'actual': y_test, 'predicted': y_pred_test})
    r2_scores = {}
    for subfield in results['subfield'].unique():
        subfield_data = results[results['subfield'] == subfield]
        r2 = r2_score(subfield_data['actual'], subfield_data['predicted'])
        r2_scores[subfield] = r2
    for subfield, r2 in r2_scores.items():
        print(f"R² score for subfield {subfield}: {r2:.3f}")
    
    # get model feature importance and plot as horizontal bar chart
    feature_importance = model.feature_importances_
    feature_importance = pd.Series(feature_importance, X.columns[0:len(X.columns)])
    feature_importance.plot(kind="barh", color="lightseagreen")
    plt.ylabel('Feature Labels')
    plt.xlabel('Feature Importances')
    plt.title('Comparison of different Feature Importances')
    plt.show()

    errors(y_test, y_pred_test, y_train, y_pred_train)

    plotting(y_test, y_train, y_pred_test, y_pred_train)


