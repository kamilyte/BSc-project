import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from database import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from paper import text as test_text


def clean_prepare_data():
    data = fetch_data()
    
    data = data.drop("all_time_hindex", axis=1)
    data = data.drop("time_bound_hindex", axis=1)
    data = data.drop("doc_doi", axis=1)
    data = data.drop("doc_title", axis=1)
    data = data.drop("text", axis=1)
    data = data.drop("doc_id", axis=1)
    
    
    
    le = LabelEncoder()
    data["quality"] = le.fit_transform(data["quality"])
    
    scaler = StandardScaler()
    data[["total_citations", "total_altmetrics", "usage", "captures", "mentions", "social_media"]] = scaler.fit_transform(data[["total_citations", "total_altmetrics", "usage", "captures", "mentions", "social_media"]])
    print(data)
    
    X_train, X_test, y_train, y_test = train_test_split(data.drop("quality", axis=1), data["quality"], test_size=0.3, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train) 
    
    print(X_test)
    print(X_train)
    y_pred = model.predict(X_test)

    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print('Precision: {:.2f}'.format(precision_score(y_test, y_pred)))
    print('Recall: {:.2f}'.format(recall_score(y_test, y_pred)))
    print('F1 Score: {:.2f}'.format(f1_score(y_test, y_pred)))

#clean_prepare_data()

    
    
