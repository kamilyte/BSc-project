import sys
sys.path.append('/Users/kamile/Desktop/Bachelor-Project/BSc-project')
import psycopg2
from data.config import DB_NAME, DB_HOST, DB_USER, DB_PASS, DB_PORT, API_KEY, SCOPUS_API_KEY, SCOPUS_BASE_URL, SCOPUS_TOKEN, PLUMX_BASE_URL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_db():
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        df = pd.read_sql_query("""SELECT * FROM scopus_database WHERE max_h_index IS NOT NULL AND text IS NOT NULL;""",conn)
        
        
        conn.commit()
        
    except Exception as e:
        print("Error creating table: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")
            
    return df

def normalise_citations(df):
    df["normalised_citations"] = 0
    df["normalised_first_year"] = 0
    df["normalised_second_year"] = 0
    citations_list = df["total_citations"]
    first_year_list = df["first_year_citations"]
    second_year_list = df["second_year_citations"]
    age_list = df["age"]
    
    for idx, (citations, first_year, second_year, age) in enumerate(zip(citations_list, first_year_list, second_year_list, age_list)):
        if citations == 0 or age == 0:
            continue
        
        normalised_citations = citations / age
        normalised_first_year = first_year / normalised_citations
        normalised_second_year = second_year / normalised_citations
        df.at[idx, "normalised_citations"] = normalised_citations
        df.at[idx, "normalised_first_year"] = normalised_first_year
        df.at[idx, "normalised_second_year"] = normalised_second_year
        
    
    return df

def data_cleaning(df):
    df = normalise_citations(df)
    df = df.drop(["doi", "title", "limited_citations", "age", "text"], axis=1)
    
    return df

def panda_to_csv(df):
    df.to_csv("cleaned_data_v3.csv", index=False)


def start_func():
    df = fetch_db()
    df = data_cleaning(df)
    panda_to_csv(df)
    print(df.head())
    
def statistics_summary():
    df = fetch_db()
    df = normalise_citations(df)
    print(df.describe().T)
    
def normalised_data():
    df = fetch_db()
    df = normalise_citations(df)
    return df
    
#statistics_summary()
    

