import psycopg2
from config import DB_NAME, DB_HOST, DB_USER, DB_PASS, DB_PORT, API_KEY, SCOPUS_API_KEY, SCOPUS_BASE_URL, SCOPUS_TOKEN
import requests
import pandas as pd
from serpapi import GoogleSearch
import os
import re
import text_extraction
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('all')

analyzer = SentimentIntensityAnalyzer()

query_list = ["cryptography and security", "data structures and algorithms", "distributed, parallel and cluster computing",  "data science and analytics", "computer vision and pattern recognition", "software engineering", "internet of things and embedded systems", "computational complexity", "graphics", "machine learning"]


def preprocess_text(text):
    # tokenize the text
    tokens = word_tokenize(text.lower())

    # remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

def sentiment_analyser(text, doi):
    # preprocess text
    text = preprocess_text(text)
    
    # sentiment analysis
    scores = analyzer.polarity_scores(text)
    
    # normalise scores
    total = scores["neg"] + scores["neu"] + scores["pos"]
    
    # insert into database
    insert_sentiment(scores["neg"] / total, scores["neu"] / total, scores["pos"] / total, scores["compound"], doi)
    
def insert_sentiment(neg_sentiment, neu_sentiment, pos_sentiment, compound_sentiment, doi):
    data = (neg_sentiment, neu_sentiment, pos_sentiment, compound_sentiment, doi)
    
    # connect to database and update table
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    UPDATE scopus_database_v4
                    SET neg_sentiment = %s,
                    neu_sentiment = %s,
                    pos_sentiment = %s,
                    compound_sentiment = %s
                    WHERE doi = %s
                    ;
                    """, data)
        
        conn.commit()
        
    except Exception as e:
        print("Error updating table: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")

# perform sentiment analysis on each doi          
def sentiment_analysis():
    df = fetch_db()
    doi_list = df["doi"]
    text_list = df["text"]
    
    for doi, text in zip(doi_list, text_list):
        sentiment_analyser(text, doi)

# create table
def database_init():
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS scopus_database_v4(
                    doi TEXT NOT NULL PRIMARY KEY,
                    title TEXT,
                    year INTEGER,
                    query TEXT,
                    text TEXT,
                    flesch_reading_ease REAL,
                    flesch_kincaid_grade REAL,
                    gunning_fog_index REAL,
                    smog_index REAL,
                    automated_readability_index REAL,
                    coleman_liau_index REAL,
                    cohesion REAL,
                    syntax REAL,
                    vocabulary REAL,
                    phraseology REAL,
                    grammar REAL,
                    conventions REAL,
                    max_hindex INTEGER,
                    avg_hindex REAL,
                    third_year_total INTEGER,
                    num_authors INTEGER,
                    combined_grade REAL
                    );
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

# fetch all data from table
def fetch_db(): 
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        df = pd.read_sql_query("""
                               SELECT * FROM scopus_database_v4;
                               """,conn)
        
        conn.commit()
        
    except Exception as e:
        print("Error fetching data from scopus_database_v4: ", e)
    finally:
        if conn:
            conn.close()
            print("Database connection closed")
            
    return df
        
# get h index of all authors       
def author_retrieval_api(author_list):
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": SCOPUS_API_KEY,
        "X-ELS-Insttoken": SCOPUS_TOKEN
    }
    
    hindex_list = []
    
    for author in author_list:
        # get author id
        author_id = author["authid"]
        
        # api access to author data
        url = f"https://api.elsevier.com/content/author/author_id/{author_id}?view=METRICS"
        response = requests.get(url, headers=headers)
    
        if response.status_code == 200:
            data = response.json()
            hindex = int(data["author-retrieval-response"][0]["h-index"])
            hindex_list.append(hindex)
            
            # prevent making too many API calls at a time
            time.sleep(0.5)
        else:
            print(f"Error: {response.text}")
            continue
        
    return hindex_list

# get pdfs of papers
def google_search_pdf(title):
    # get paper using title
    params = {
        "engine": "google_scholar",
        "q": title,
        "api_key": API_KEY,
        "hl":"en",
        "start": 0,
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    search_status = results["search_metadata"]["status"]
    if search_status == "Error":
        print("Error: unsuccessful retrieval")
        return False
        
    try:
        organic_results = results["organic_results"]
        result = organic_results[0]
        format = result["resources"][0]["file_format"]
        
        # if PDF doesnt exist return as False
        if format != "PDF":
            return False
        
        # set name for pdf
        url = result["resources"][0]["link"]
        pattern = re.compile('[^a-zA-Z ]')
        cleaned_title = pattern.sub('', title)
        filename = cleaned_title + ".pdf"
        folder_path = "papers"
        response = requests.get(url, stream=True, timeout=(60, 300))
            
    except:
        print("Error: problem with retrieving/downloading pdf ")
        return False
    
    # dowload pdf
    if response.status_code == 200:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, "wb") as pdf_object:
                pdf_object.write(response.content)
                print("Download successful") 
        except:
            print("Took too long to download")
            return False
    else:
        print("Download unsuccessful")
        return False
        
    return True

# get maximum h-index from list
def max_h_index(h_index_list):
    if not h_index_list:
        return 0
    return max(h_index_list)

# get minimum h-index from list
def min_h_index(h_index_list):
    if not h_index_list:
        return 0
    return min(h_index_list)

# get average h-index of list
def average_h_index(h_index_list):
    if not h_index_list:
        return 0.0
    return sum(h_index_list) / len(h_index_list)


def insert_data(doi, title, year, query, text, max_hindex, avg_hindex, third_year_total, total):
    data = (doi, title, year, query, text, max_hindex, avg_hindex, third_year_total, total)
    
    # insert data into database
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    INSERT INTO scopus_database_v4(doi, title, year, query, text, max_hindex, avg_hindex, third_year_total, total)
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """, data)
        
        conn.commit()
        
    except Exception as e:
        print("Error inserting into database: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")
            
def insert_author_data(doi, max_hindex, avg_hindex, num_authors, issn, min_hindex):
    data = (max_hindex, avg_hindex, num_authors, issn, min_hindex, doi)
    
    # insert author data
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    UPDATE scopus_database_v4
                    SET max_hindex = %s,
                    avg_hindex = %s,
                    num_authors = %s,
                    issn = %s,
                    min_hindex = %s
                    WHERE doi = %s
                    ;
                    """, data)
        

        conn.commit()
        
    except Exception as e:
        print("Error updating table: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")
   

# get citation data of paper after three years
def citation_overview_api(doi, year):
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": SCOPUS_API_KEY,
        "X-ELS-Insttoken": SCOPUS_TOKEN
    }
    # year of interest
    end_year = year + 3
    
    # API call
    url = f"https://api.elsevier.com/content/abstract/citations?doi={doi}&date={year}-{end_year}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        try:
            entry = data["abstract-citations-response"]
            summary = entry["citeColumnTotalXML"]["citeCountHeader"]
            third_year_total = int(summary["rangeColumnTotal"])
        except:
            print("Unable to get citation overview data")
            return None
    else:
        print(f"Error: {response.text}")
        return None
    
    return third_year_total
    
# retrieve papers of interest
def scopus_search_api(query):
    count = 200
    
    # fields of interest
    field = "dc:title,citedby-count,prism:coverDate,prism:doi,author"
    
    start = 0
    
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": SCOPUS_API_KEY,
        "X-ELS-Insttoken": SCOPUS_TOKEN
    }
    
    # make API call
    url = f"{SCOPUS_BASE_URL}?query={query}&count={count}&date=2017-2020&subj=COMP&start={start}&field={field}&sort=-citedby-count"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        
        for entry in data["search-results"]["entry"]:
            try:
                doi = entry["prism:doi"]
                title = entry["dc:title"]
                date = entry["prism:coverDate"]
                year = int(date.split("-")[0])
                total = int(entry["citedby-count"])
                
                # if no PDF exists, ignore
                if not google_search_pdf(title):
                    continue
                
                pattern = re.compile('[^a-zA-Z ]')
                cleaned_title = pattern.sub('', title)
                filename = cleaned_title + ".pdf"
                
                text = text_extraction.get_text(filename)
                
                # if failure to retrieve text, ignore
                if not text:
                    continue
                
                hindex_list = author_retrieval_api(entry["author"])
                max_hindex = max_h_index(hindex_list)
                avg_hindex = average_h_index(hindex_list)

                third_year_total = citation_overview_api(doi, year)

                # insert into table
                insert_data(doi, title, year, query, text, max_hindex, avg_hindex, third_year_total, total)
                
            except Exception as e:
                print("Exception encountered: ", e)
        
    else:
        print(f"Error: {response.text}")
        return

# get author h-index data and insert into table
def scopus_author_api(doi):
    
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": SCOPUS_API_KEY,
        "X-ELS-Insttoken": SCOPUS_TOKEN
    }
    field = "dc:title,citedby-count,prism:coverDate,prism:doi,author,prism:issn"
    
    url = f"{SCOPUS_BASE_URL}?query={doi}&field={field}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        
        entry = data["search-results"]["entry"][0]
                
        try:
            issn = entry["prism:issn"]
            
        except Exception as e:
            issn = None
                
        hindex_list = author_retrieval_api(entry["author"])
        max_hindex = max_h_index(hindex_list)
        avg_hindex = average_h_index(hindex_list)
        min_hindex = min_h_index(hindex_list)
        num_authors = len(hindex_list)
            
        insert_author_data(doi, max_hindex, avg_hindex, num_authors, issn, min_hindex)
                
            
    else:
        print(f"Error: {response.text}")
        return

# loop through all doi to get h-index data   
def get_hindex_data():
    df = fetch_db()
    doi_list = df["doi"]
    
    for doi in doi_list:
        scopus_author_api(doi)
    
# set query labels for each query made        
def add_labels():
    
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    UPDATE scopus_database_v4
                    SET query_label = 0
                    WHERE query = 'cryptography and security'
                    ;
                    """)
        
        cur.execute("""
                    UPDATE scopus_database_v4
                    SET query_label = 1
                    WHERE query = 'data structures and algorithms'
                    ;
                    """)
        
        cur.execute("""
                    UPDATE scopus_database_v4
                    SET query_label = 2
                    WHERE query = 'distributed, parallel and cluster computing'
                    ;
                    """)
        
        cur.execute("""
                    UPDATE scopus_database_v4
                    SET query_label = 3
                    WHERE query = 'data science and analytics'
                    ;
                    """)
        
        cur.execute("""
                    UPDATE scopus_database_v4
                    SET query_label = 4
                    WHERE query = 'computer vision and pattern recognition'
                    ;
                    """)
        
        cur.execute("""
                    UPDATE scopus_database_v4
                    SET query_label = 5
                    WHERE query = 'software engineering'
                    ;
                    """)
        
        cur.execute("""
                    UPDATE scopus_database_v4
                    SET query_label = 6
                    WHERE query = 'internet of things and embedded systems'
                    ;
                    """)
        
        cur.execute("""
                    UPDATE scopus_database_v4
                    SET query_label = 7
                    WHERE query = 'computational complexity'
                    ;
                    """)
        
        cur.execute("""
                    UPDATE scopus_database_v4
                    SET query_label = 8
                    WHERE query = 'graphics'
                    ;
                    """)
        
        cur.execute("""
                    UPDATE scopus_database_v4
                    SET query_label = 9
                    WHERE query = 'machine learning'
                    ;
                    """)
        
        conn.commit()
        
    except Exception as e:
        print("Error updating table: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")
    
# set the combined grades
def combine():
    
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    UPDATE scopus_database_v4
                    SET combined_grade = (flesch_kincaid_grade + gunning_fog_index + smog_index + automated_readability_index + coleman_liau_index) / 5
                    ;
                    """)

        conn.commit()
        
    except Exception as e:
        print("Error updating table: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")
    




    




    


