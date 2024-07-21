import psycopg2
from data.config import DB_NAME, DB_HOST, DB_USER, DB_PASS, DB_PORT
import textstat

# updating table with readability metrics
def add_readability_metrics(Flesch_Reading_Ease, Flesch_Kincaid_Grade, Gunning_Fog_Index, SMOG_Index, Automated_Readability_Index, Coleman_Liau_Index, doi):
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    UPDATE scopus_database_v4
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
            
# computing the readability of text using textstat library
def compute_readability_metrics(text, doi):
    Flesch_Reading_Ease = textstat.flesch_reading_ease(text)
    Flesch_Kincaid_Grade = textstat.flesch_kincaid_grade(text)
    Gunning_Fog_Index = textstat.gunning_fog(text)
    SMOG_Index = textstat.smog_index(text)
    Automated_Readability_Index = textstat.automated_readability_index(text)
    Coleman_Liau_Index = textstat.coleman_liau_index(text)
    
    add_readability_metrics(Flesch_Reading_Ease, Flesch_Kincaid_Grade, Gunning_Fog_Index, SMOG_Index, Automated_Readability_Index, Coleman_Liau_Index, doi)


        
