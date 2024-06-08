from serpapi import GoogleSearch
from config import API_KEY, SCOPUS_API_KEY, SCOPUS_BASE_URL, SCOPUS_TOKEN, PLUMX_BASE_URL
import requests
import os
from database import db_insert
from text_extraction import get_text


def download_pdf_file(url, filename) -> bool:
    folder_path = "data/test_papers"
    
    try:
        response = requests.get(url, stream=True)
    except Exception as e:
        print("Response failed: ", e)
        return False
    if response.status_code == 200:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "wb") as pdf_object:
            pdf_object.write(response.content)
            print("Download successful")
            return True
    else:
        print("Download unsuccessful")
        return False
      
      
def calculate_h_index(citation_list):
    citation_list.sort(reverse=True)
    h_index = 0
    for idx, citation in enumerate(citation_list, start=1):
        if citation >= idx:
            h_index = idx  
        else:
            break
    return h_index

        

def scopus_search_api(query):
    # retrieve top cited articles under query
    count = 10
    url = f"{SCOPUS_BASE_URL}?query={query}&count={count}&date=2017-2022&sort=+citedby-count"
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": SCOPUS_API_KEY,
        "X-ELS-Insttoken": SCOPUS_TOKEN
    }
    response = requests.get(url, headers=headers)
    
    article_list = []    # to store article titles to use for Google Scholar search
    article_doi_list = []    # to store article DOI to use for PlumX altmetrics search
    
    if response.status_code == 200:
        data = response.json()
        
        for entry in data["search-results"]["entry"]:
            exception_encountered = False
            
            # get DOI
            try:
                doi = entry["prism:doi"]
            except:
                exception_encountered = True 
            
            # get title and append to list
            if not exception_encountered:
                title = entry["dc:title"]
                article_list.append(title)
                article_doi_list.append(doi)
                
        return article_list, article_doi_list
    else:
        print(f"Error: {response.text}")
        
        
def retrieve_altmetrics(data):
    total_altmetrics = 0
    usage = 0
    captures = 0
    mentions = 0
    social_media = 0
    
    if "count_categories" in data:
        for category in data["count_categories"]:
            if category["name"] == "capture":
                captures = category["total"]
            elif category["name"] == "mention":
                mentions = category["total"]
            elif category["name"] == "usage":
                usage = category["total"]
            elif category["name"] == "socialMedia":
                social_media = category["total"]
            
    total_altmetrics += usage + captures + mentions + social_media
    return total_altmetrics, usage, captures, mentions, social_media


def google_scholar_api(query):
    article_list, article_doi_list = scopus_search_api(query)
    
    # for each title retrieve: 
    # - total citations
    # - h_index of authors across all time 
    # - h_index of authors across bounded time 
    # - total altmetrics 
    # - usage 
    # - captures 
    # - mentions 
    # - social media
    for title, doi in zip(article_list[4:], article_doi_list[4:]):
        
        # parameters for Google Scholar API search
        params = {
            "engine": "google_scholar",
            "q": title,
            "api_key": API_KEY,
            "hl":"en",
            "start": 0,
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # check if no error occurred
        search_status = results["search_metadata"]["status"]
        if search_status == "Error":
            print("Error: unsuccessful retrieval")
            return None 
        
        organic_results = results["organic_results"]
        result = organic_results[0]
        
        # check if PDF exists
        if "resources" in result and result["resources"][0]["file_format"] == "PDF":
            
            # retrieve PlumX altmetrics 
            plum_url = f"{PLUMX_BASE_URL}/doi/{doi}"
            headers = {
                "Accept": "application/json",
                "X-ELS-APIKey": SCOPUS_API_KEY,
                "X-ELS-Insttoken": SCOPUS_TOKEN
            }
            response = requests.get(plum_url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                total_altmetrics, usage, captures, mentions, social_media = retrieve_altmetrics(data)
            else:
                print(f"Error: {response.text}")
                continue
            
            
            # download PDF
            url = result["resources"][0]["link"]
            filename = title + ".pdf"
            download_success = download_pdf_file(url, filename)
            if not download_success:
                continue
            
            total_citations = result["inline_links"]["cited_by"]["total"]   # retrieve total citations for article 
            
            # retrieve h_index of authors available on Google Scholar
            if "publication_info" not in result or "authors" not in result["publication_info"]:
                continue
            
            authors = result["publication_info"]["authors"]
            authorID_list = []     # list of author ID for Google Scholar Author search
            all_time_h_index_list = []      # list of h_index across all time of authors 
            time_bound_h_index_list = []    # list of h_index of authors within 2017 - 2022 
            
            for author in authors:
                authorID_list.append(author["author_id"])
                
            for authorID in authorID_list:
                time_bound_citations = []    # list of citations made to author within 2017 - 2022 
                
                # author params
                author_params = {
                    "engine": "google_scholar_author",
                    "author_id": authorID,
                    "api_key": API_KEY
                }
                author_search = GoogleSearch(author_params)
                author_results = author_search.get_dict()
                
                all_time_h_index = author_results["cited_by"]["table"][1]["h_index"]["all"]   # all time h_index of author
                
                # citations between 2017-2022
                for entry in author_results["cited_by"]["graph"]:
                    year = entry["year"]
                    if year >= 2017 and year <= 2022:
                        time_bound_citations.append(entry["citations"])
                
                time_bound_h_index = calculate_h_index(time_bound_citations)    # calculate h_index between 2017 - 2022
                
                all_time_h_index_list.append(all_time_h_index)
                time_bound_h_index_list.append(time_bound_h_index)
                
            
            text = get_text(filename)
            
            db_insert(doi, title, text, total_citations, all_time_h_index_list, time_bound_h_index_list, total_altmetrics, usage, captures, mentions, social_media)
            



def scopus_search_api_2(query):
    # retrieve top cited articles under query
    count = 10
    url = f"{SCOPUS_BASE_URL}?query={query}&count={count}&date=2017-2022&sort=+citedby-count"
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": SCOPUS_API_KEY,
        "X-ELS-Insttoken": SCOPUS_TOKEN
    }
    response = requests.get(url, headers=headers)
    
    article_list = []    # to store article titles to use for Google Scholar search
    article_doi_list = []    # to store article DOI to use for PlumX altmetrics search
    
    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print(f"Error: {response.text}")
            


def scopus_search_api_3():
    # retrieve top cited articles under query
    count = 10
    url = f"https://api.elsevier.com/content/abstract/citations?doi=10.1109/TPAMI.2016.2577031&date=2017-2018"
    headers = {
        "Accept": "application/json",
        "X-ELS-APIKey": SCOPUS_API_KEY,
        "X-ELS-Insttoken": SCOPUS_TOKEN
    }
    response = requests.get(url, headers=headers)
    
    article_list = []    # to store article titles to use for Google Scholar search
    article_doi_list = []    # to store article DOI to use for PlumX altmetrics search
    
    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print(f"Error: {response.text}")
        
    # query = "information retrieval"
    # count = 200
    
    # # get high impact
    # url = f"{SCOPUS_BASE_URL}?query={query}&count={count}&date=2017-2022&sort=-citedby-count&subj=COMP"
    # headers = {
    #     "Accept": "application/json",
    #     "X-ELS-APIKey": SCOPUS_API_KEY,
    #     "X-ELS-Insttoken": SCOPUS_TOKEN
    # }
    # response = requests.get(url, headers=headers)
    
    # tuple_list = []
    
    # if response.status_code == 200:
    #     data = response.json()
    #     print("okay")
    # else:
    #     print(f"Error: {response.text}")
    #     return
        
scopus_search_api_3()

