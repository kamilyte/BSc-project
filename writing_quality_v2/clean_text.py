import sys
sys.path.append('/Users/kamile/Desktop/Bachelor-Project/BSc-project')
from paper import text
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
import re
import pandas as pd
import psycopg2
from data.config import DB_NAME, DB_HOST, DB_USER, DB_PASS, DB_PORT, API_KEY, SCOPUS_API_KEY, SCOPUS_BASE_URL, SCOPUS_TOKEN, PLUMX_BASE_URL

def fetch_db():
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        
        
        # cur.execute("""
        #             ALTER TABLE scopus_database
        #             ADD cohesion REAL;
        #             """)
        
        # cur.execute("""
        #             ALTER TABLE scopus_database
        #             ADD syntax REAL;
        #             """)
        
        # cur.execute("""
        #             ALTER TABLE scopus_database
        #             ADD vocabulary REAL;
        #             """)
        
        # cur.execute("""
        #             ALTER TABLE scopus_database
        #             ADD phraseology REAL;
        #             """)
        
        # cur.execute("""
        #             ALTER TABLE scopus_database
        #             ADD grammar REAL;
        #             """)
        
        # cur.execute("""
        #             ALTER TABLE scopus_database
        #             ADD conventions REAL;
        #             """)
        
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

def replace_unnecessary_chars(text):
    cleaned_text = text.replace('- ', '')
    pattern = r'[^a-zA-Z\s().]'
    cleaned_text = re.sub(pattern, '', cleaned_text)
    cleaned_text = cleaned_text.replace('.  .', '.')
    return cleaned_text

def preprocess_text(text):
    text = replace_unnecessary_chars(text)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words

# print(replace_unnecessary_chars(text))
# print(preprocess_text(text))
#print(fetch_db())
