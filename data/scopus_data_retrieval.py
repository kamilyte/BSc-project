import psycopg2
from config import DB_NAME, DB_HOST, DB_USER, DB_PASS, DB_PORT, API_KEY, SCOPUS_API_KEY, SCOPUS_BASE_URL, SCOPUS_TOKEN, PLUMX_BASE_URL
import requests
import pandas as pd
from serpapi import GoogleSearch
import os
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

query_list = ["machine learning", "cryptography and security", "data structures and algorithms", "distributed, parallel and cluster computing", "information retrieval", "computer vision and pattern recognition", "software engineering", "internet of things and embedded systems", "computational complexity", "graphics"]


# retry_strategy = Retry(
#     total=3,  # Total number of retries
#     backoff_factor=1,  # A delay factor for retries
#     status_forcelist=[500, 502, 503, 504],  # Retry for these HTTP status codes
#     method_whitelist=["HEAD", "GET", "OPTIONS"]  # Methods to retry
# )

# session = requests.Session()
# adapter = HTTPAdapter(max_retries=retry_strategy)
# session.mount("https://", adapter)
# session.mount("http://", adapter)


def database_init():
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        # cur.execute("""CREATE TABLE IF NOT EXISTS scopus_database(
        #             doi TEXT NOT NULL PRIMARY KEY,
        #             title TEXT NOT NULL,
        #             year INTEGER NOT NULL,
        #             query TEXT NOT NULL,
        #             total_citations INTEGER NOT NULL,
        #             limited_citations INTEGER NOT NULL,
        #             h_index INTEGER NOT NULL,
        #             usage INTEGER NOT NULL,
        #             captures INTEGER NOT NULL,
        #             mentions INTEGER NOT NULL,
        #             social_media INTEGER NOT NULL);
        #     """)
        
        # cur.execute(""" ALTER TABLE scopus_database
        #                 ADD impact VARCHAR(5) NOT NULL;""")
        
        # cur.execute("""
        #             ALTER TABLE scopus_database
        #             ALTER COLUMN limited_citations DROP NOT NULL;
        #             """)

        # cur.execute("""
        #             ALTER TABLE scopus_database
        #             ALTER COLUMN h_index DROP NOT NULL;
        #             """)
        
        # cur.execute(""" ALTER TABLE scopus_database
        #                 ADD second_year_citations INTEGER;""")
        
        # cur.execute("""
        #             SELECT * FROM scopus_database
        #             WHERE doi = "10.1109/CVPR42600.2020.00813";
        #             """)
        
        # cur.execute("""
        #             UPDATE scopus_database
        #             SET age = 2024 - year
        #             WHERE age IS NULL;
        #             """)
        
        cur.execute("""
                    ALTER TABLE scopus_database
                    RENAME COLUMN h_index TO avg_h_index;
                    """)
        
        cur.execute(""" ALTER TABLE scopus_database
                        ADD max_h_index INTEGER;""")
        
        cur.execute("""
                    ALTER TABLE scopus_database
                    ADD text TEXT;
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
            

def get_doi_titles(query):
    count = 200
    start = 200
    
    # get high impact
    url = f"{SCOPUS_BASE_URL}?query={query}&count={count}&date=2017-2022&sort=-citedby-count&subj=COMP&start={start}"
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": SCOPUS_API_KEY,
        "X-ELS-Insttoken": SCOPUS_TOKEN
    }
    response = requests.get(url, headers=headers)
    
    tuple_list = []
    
    if response.status_code == 200:
        data = response.json()
        
        for entry in data["search-results"]["entry"]:
            exception_encountered = False
            
            # get DOI
            try:
                doi = entry["prism:doi"]
                cited_by_count = entry["citedby-count"]
                date = entry["prism:coverDate"]
                year = date.split("-")[0]
            except:
                exception_encountered = True 
                #print("Error: Unable to get DOI/cited_by_count of high impact paper")
            
            # get title and append to list
            if not exception_encountered:
                title = entry["dc:title"]
                tuple_list.append((title, doi, year, cited_by_count, "high"))
    else:
        print(f"Error: {response.text}")
        return
        
    # get low impact
    url = f"{SCOPUS_BASE_URL}?query={query}&count={count}&date=2017-2022&sort=+citedby-count&subj=COMP&start={start}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        
        for entry in data["search-results"]["entry"]:
            exception_encountered = False
            
            # get DOI
            try:
                doi = entry["prism:doi"]
                cited_by_count = entry["citedby-count"]
                date = entry["prism:coverDate"]
                year = date.split("-")[0]
            except:
                exception_encountered = True 
                #print("Error: Unable to get DOI/cited_by_count of low impact paper")
            
            # get title and append to list
            if not exception_encountered:
                title = entry["dc:title"]
                tuple_list.append((title, doi, year, cited_by_count, "low"))
    else:
        print(f"Error: {response.text}")
        return
        
    return tuple_list

def delete_entry(doi):
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    DELETE FROM scopus_database
                    WHERE doi = %s;
                    """, (doi,))
        
        conn.commit()
        
    except Exception as e:
        print("Error deleting entry: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")
            
def change_altmetric_data(doi, usage, captures, mentions, social_media):
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    UPDATE scopus_database
                    SET usage = %s,
                        captures = %s,
                        mentions = %s,
                        social_media = %s
                    WHERE doi = %s;
                    """, (usage, captures, mentions, social_media, doi))
        
        
        
        conn.commit()
        
    except Exception as e:
        print("Error updating altmetric data: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")

def alter_altmetrics():
    df = fetch_db()
    headers = {
            "Accept": "application/json",
            "X-ELS-APIKey": SCOPUS_API_KEY,
            "X-ELS-Insttoken": SCOPUS_TOKEN
        }
    
    for doi in df["doi"]:
        url = f"{PLUMX_BASE_URL}/doi/{doi}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            usage = 0
            captures = 0
            mentions = 0
            social_media = 0
            
            try: 
                for category in data["count_categories"]:
                    if category["name"] == "capture":
                        captures = category["total"]
                    elif category["name"] == "mention":
                        mentions = category["total"]
                    elif category["name"] == "usage":
                        usage = category["total"]
                    elif category["name"] == "socialMedia":
                        social_media = category["total"]
            except:
                print("Error: unable to get altmetric data")
                print(data)
                delete_entry(doi)
                
            
        else:
            print(f"Error: {response.text}")
            return
            

def get_altmetrics(tuple_list):
    altmetric_list = []
    for tuple in tuple_list:
        doi = tuple[1]
        url = f"{PLUMX_BASE_URL}/doi/{doi}"
        headers = {
            "Accept": "application/json",
            "X-ELS-APIKey": SCOPUS_API_KEY,
            "X-ELS-Insttoken": SCOPUS_TOKEN
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            usage = 0
            captures = 0
            mentions = 0
            social_media = 0
            
            try: 
                for category in data["count_categories"]:
                    if category["name"] == "capture":
                        captures = category["total"]
                    elif category["name"] == "mention":
                        mentions = category["total"]
                    elif category["name"] == "usage":
                        usage = category["total"]
                    elif category["name"] == "socialMedia":
                        social_media = category["total"]
            except:
                print("Error: unable to get altmetric data")
                
            altmetric_list.append((usage, captures, mentions, social_media))
            
        else:
            print(f"Error: {response.text}")
            return

    return altmetric_list


def init_insert(doi, title, year, query, total_citations, usage, captures, mentions, social_media, impact):
    insert_data = (doi, title, year, query, total_citations, usage, captures, mentions, social_media, impact)
    
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    INSERT INTO scopus_database(doi, title, year, query, total_citations, usage, captures, mentions, social_media, impact)
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """, insert_data)

        conn.commit()
        
    except Exception as e:
        print("Error inserting into scopus_database: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")
            
def fetch_db(): 
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
  
        df = pd.read_sql_query("""SELECT * FROM scopus_database WHERE max_h_index IS NOT NULL AND text IS NOT NULL;""",conn)
        
        conn.commit()
        
    except Exception as e:
        print("Error fetching data from train_database: ", e)
    finally:
        if conn:
            conn.close()
            print("Database connection closed")
            
    return df
          

def get_data(query_list):
    for query in query_list:
        doc_tuple_list = get_doi_titles(query)
        altmetric_tuple_list = get_altmetrics(doc_tuple_list)
        length = len(doc_tuple_list)
        for i in range(length):
            document = doc_tuple_list[i]
            doi = document[1]
            title = document[0]
            year = int(document[2])
            total_citations = int(document[3])
            impact = document[4]
            
            altmetric = altmetric_tuple_list[i]
            usage = altmetric[0]
            captures = altmetric[1]
            mentions = altmetric[2]
            social_media = altmetric[3]
            
            init_insert(doi, title, year, query, total_citations, usage, captures, mentions, social_media, impact)
        
        print()
        print(f"Query: {query}")
        print()
            
def add_limited_citations(doi, year_1_cites, year_2_cites):
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        total_cites = year_1_cites + year_2_cites
        
        cur.execute("""
                    UPDATE scopus_database
                    SET first_year_citations = %s,
                        second_year_citations = %s,
                        limited_citations = %s
                    WHERE doi = %s;
                    """, (year_1_cites, year_2_cites, total_cites, doi))
        
        conn.commit()
        
    except Exception as e:
        print("Error updating: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")
    
    
        
def populate_limited_citations():
    df = fetch_db()
    doi_list = df["doi"]
    year_list = df["year"]
    length = len(doi_list)
    
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": SCOPUS_API_KEY,
        "X-ELS-Insttoken": SCOPUS_TOKEN
    }
    
    for i in range(length):
        doi = doi_list[i]
        year_1 = year_list[i]
        year_2 = year_1 + 1
        url = f"https://api.elsevier.com/content/abstract/citations?doi={doi}&date={year_1}-{year_2}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            try:
                entry = data["abstract-citations-response"]
                summary = entry["citeColumnTotalXML"]["citeCountHeader"]
                citation_column = summary["columnTotal"]
                year_1_cites = int(citation_column[0]["$"])
                year_2_cites = int(citation_column[1]["$"])
                add_limited_citations(doi, year_1_cites, year_2_cites)
                
            except:
                print("Unable to get citation overview data")
        else:
            print(f"Error: {response.text}")
            
            
def max_h_index(h_index_list):
    if not h_index_list:
        return 0
    return max(h_index_list)

def average_h_index(h_index_list):
    if not h_index_list:
        return 0.0
    return sum(h_index_list) / len(h_index_list)

def insert_h_index(doi, max_h_index, avg_h_index):
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    UPDATE scopus_database
                    SET max_h_index = %s,
                        avg_h_index = %s
                    WHERE doi = %s;
                    """, (max_h_index, avg_h_index, doi))
        
        conn.commit()
        
    except Exception as e:
        print("Error updating h_index: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")

            
def google_search():
    df = fetch_db()
    title_list = df["title"]
    doi_list = df["doi"]
    
   
    
    for doi, title in zip(doi_list, title_list):
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
            delete_entry(doi)
            continue
        
        
        try:
            organic_results = results["organic_results"]
            authors = organic_results[0]["publication_info"]["authors"]
            authorID_list = []
            h_index_list = []
            
            for author in authors:
                authorID_list.append(author["author_id"])
                
            for authorID in authorID_list:
                author_params = {
                    "engine": "google_scholar_author",
                    "author_id": authorID,
                    "api_key": API_KEY
                }
                
                author_search = GoogleSearch(author_params)
                author_results = author_search.get_dict()
                h_index = author_results["cited_by"]["table"][1]["h_index"]["all"] 
                h_index_list.append(h_index)
        except Exception as e:
                print("Error: with author details: ", e)
                delete_entry(doi)
                continue
        
        try:
            organic_results = results["organic_results"]
            result = organic_results[0]
            url = result["resources"][0]["link"]
            pattern = re.compile('[^a-zA-Z ]')
            cleaned_title = pattern.sub('', title)
            
            filename = cleaned_title + ".pdf"
            folder_path = "research_papers_v2"
            response = requests.get(url, stream=True, timeout=(20, 200))
        except Exception as e:
            print("Error: problem with downloading pdf ", e)
            delete_entry(doi)
            continue
        
        if response.status_code == 200:
            filepath = os.path.join(folder_path, filename)
            
            try:
                with open(filepath, "wb") as pdf_object:
                    pdf_object.write(response.content)
                    print("Download successful") 
            except:
                print("Took too long to download")
                delete_entry(doi)
                continue
                    
        else:
            print("Download unsuccessful")
            delete_entry(doi)
            continue
        
        
        max = max_h_index(h_index_list)
        avg = average_h_index(h_index_list)
        insert_h_index(doi, max, avg)
        
        
        
        
        
    
    
    

#database_init()
#get_data(query_list)
#populate_limited_citations()
#alter_altmetrics()
#google_search()
print(fetch_db())




    


