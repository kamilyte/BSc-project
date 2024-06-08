import sys
sys.path.append('/Users/kamile/Desktop/Bachelor-Project/BSc-project')
import psycopg2
from data.config import DB_NAME, DB_HOST, DB_USER, DB_PASS, DB_PORT, API_KEY, SCOPUS_API_KEY, SCOPUS_BASE_URL, SCOPUS_TOKEN, PLUMX_BASE_URL
import textstat
import pandas as pd
from paper import text
import clean_text

def add_readability_metrics(Flesch_Reading_Ease, Flesch_Kincaid_Grade, Gunning_Fog_Index, SMOG_Index, Automated_Readability_Index, Coleman_Liau_Index, doi):
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    UPDATE scopus_database
                    SET flesch_reading_ease = %s,
                        flesch_kincaid_grade = %s,
                        gunning_fog_index = %s,
                        smog_index = %s,
                        automated_readability_index = %s,
                        coleman_liau_index = %s
                    WHERE doi = %s;
                    """, (Flesch_Reading_Ease, Flesch_Kincaid_Grade, Gunning_Fog_Index, SMOG_Index, Automated_Readability_Index, Coleman_Liau_Index, doi))
        
        conn.commit()
        
    except Exception as e:
        print("Error updating: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")
            

def compute_readability_metrics(text, doi):
    Flesch_Reading_Ease = textstat.flesch_reading_ease(text)
    Flesch_Kincaid_Grade = textstat.flesch_kincaid_grade(text)
    Gunning_Fog_Index = textstat.gunning_fog(text)
    SMOG_Index = textstat.smog_index(text)
    Automated_Readability_Index = textstat.automated_readability_index(text)
    Coleman_Liau_Index = textstat.coleman_liau_index(text)
    
    add_readability_metrics(Flesch_Reading_Ease, Flesch_Kincaid_Grade, Gunning_Fog_Index, SMOG_Index, Automated_Readability_Index, Coleman_Liau_Index, doi)


def readability():
    df = clean_text.fetch_db()
    doi_list = df["doi"]
    text_list = df["text"]
    
    for doi, text in zip(doi_list, text_list):
        text = clean_text.replace_unnecessary_chars(text)
        compute_readability_metrics(text, doi)
        
    

#print(compute_readability_metrics(text))
readability()
print(clean_text.fetch_db())