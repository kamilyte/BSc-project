import psycopg2
import pandas as pd
from .config import DB_NAME, DB_HOST, DB_USER, DB_PASS, DB_PORT


def database_init():
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS train_database(
                    doc_id SERIAL PRIMARY KEY,
                    doc_doi TEXT NOT NULL,
                    doc_title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    total_citations INTEGER NOT NULL,
                    all_time_hindex INTEGER[] NOT NULL,
                    time_bound_hindex INTEGER[] NOT NULL,
                    total_altmetrics INTEGER NOT NULL,
                    usage INTEGER,
                    captures INTEGER,
                    mentions INTEGER,
                    social_media INTEGER);
            """)
        
        conn.commit()
        
    except Exception as e:
        print("Error creating table: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")
            
def db_insert(doi, title, text, citations, all_h_index, time_h_index, altmetrics, usage, captures, mentions, social_media):
    quality = "low"
    insert_data = (doi, title, text, citations, all_h_index, time_h_index, altmetrics, usage, captures, mentions, social_media, quality)
    
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""INSERT INTO train_database(doc_doi, doc_title, text, total_citations, all_time_hindex, time_bound_hindex, total_altmetrics, usage, captures, mentions, social_media, quality)
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""", insert_data)
        
        conn.commit()
        
    except Exception as e:
        print("Error inserting into train_database: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")
            

def fetch_data():
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
  
        df = pd.read_sql_query('SELECT * FROM train_database;',conn)
        
        conn.commit()
        
    except Exception as e:
        print("Error fetching data from train_database: ", e)
    finally:
        if conn:
            conn.close()
            print("Database connection closed")
            
    return df
      
def add_column():
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        # cur.execute(""" ALTER TABLE train_database
        #                 ADD quality VARCHAR(5);""")
        
        # cur.execute(""" UPDATE train_database
        #                 SET quality = 'high';""")
        
        cur.execute(""" DELETE FROM train_database 
                    WHERE doc_id = 15;
                    """)

        
        
        conn.commit()
        
    except Exception as e:
        print("Error inserting column: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")
            

