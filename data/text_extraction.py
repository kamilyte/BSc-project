from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
import psycopg2
from config import DB_NAME, DB_HOST, DB_USER, DB_PASS, DB_PORT, API_KEY, SCOPUS_API_KEY, SCOPUS_BASE_URL, SCOPUS_TOKEN, PLUMX_BASE_URL
import pandas as pd
import re

def fetch_db(): 
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
  
        df = pd.read_sql_query("""SELECT * FROM scopus_database WHERE max_h_index IS NOT NULL AND text IS NOT NULL AND impact = 'high';""",conn)
        
        conn.commit()
        
    except Exception as e:
        print("Error fetching data from scopus_database: ", e)
    finally:
        if conn:
            conn.close()
            print("Database connection closed")
            
    return df

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

def clean_text(text):
    text = text.replace('\n', " ")
    idx = text.lower().find("abstract")
    if idx != -1:
        text = text[idx:]
        
    start = 0
    indices = []
    while start < len(text):
        start = text.lower().find("references", start)
        if start == -1:
            break
        indices.append(start)
        start += len("references")  
        
    if indices:
        idx = indices[-1]
        text = text[:idx]
    
    return text
    
    

def get_text(title):
    path = f"research_papers_v2/{title}"
    flag = True
    extracted_text = ""
    
    pdf_file = path
    
    try: 
        pages = convert_from_path(pdf_file)
    
        for page in pages:
            # Step 2: Preprocess the image (deskew)
            preprocessed_image = deskew(np.array(page))

            # Step 3: Extract text using OCR
            text = extract_text_from_image(preprocessed_image)
            extracted_text += text
            
    except Exception as e:
        print("Error extracting text: ", e)
        flag = False
        
    if flag:
        cleaned_text = clean_text(extracted_text)
        return cleaned_text
    else:
        return None

def store_text(doi, text):
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    UPDATE scopus_database
                    SET text = %s
                    WHERE doi = %s;
                    """, (text, doi))
        
        conn.commit()
        
    except Exception as e:
        print("Error updating text: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")

def get_store_text():
    df = fetch_db()
    print(df)
    doi_list = df["doi"]
    title_list = df["title"]
    
    for doi, title in zip(doi_list, title_list):
        pattern = re.compile('[^a-zA-Z ]')
        cleaned_title = pattern.sub('', title)
        filename = cleaned_title + ".pdf"
        text = get_text(filename)
        if text:
            store_text(doi, text)
        
        

#print(get_text("A comprehensive survey on support vector machine classification Applications challenges and trends.pdf"))
# get_store_text()
print(fetch_db()["text"][1])  
print(fetch_db()["doi"][1])       
