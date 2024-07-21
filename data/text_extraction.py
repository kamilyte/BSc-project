from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
import psycopg2
from config import DB_NAME, DB_HOST, DB_USER, DB_PASS, DB_PORT
import pandas as pd
import re

def clean_text(text):
    
    # replace newlines
    text = text.replace('\n', " ")
    
    # remove text before the introduction
    idx = text.lower().find("introduction")
    if idx != -1:
        text = text[idx:]
    
    # remove text after last occuring references 
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

# fetch table
def fetch_db(): 
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
  
        df = pd.read_sql_query("""
                               SELECT * FROM scopus_database_v4 
                               WHERE text IS NULL AND abstract IS NOT NULL AND pdf = 'yes';
                               """,conn)
        
        conn.commit()
        
    except Exception as e:
        print("Error fetching data from scopus_database: ", e)
    finally:
        if conn:
            conn.close()
            print("Database connection closed")
            
    return df

def deskew(image):
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    
    # get coordinates of image
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    # get angle that page should rotate by
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # rotate page
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def extract_text_from_image(image):
    # use OCR to get text from image
    text = pytesseract.image_to_string(image)
    return text

# remove header and footer data
def remove_header_footer(image, header_height=50, footer_height=50):
    height, width = image.shape[:2]
    image = image[header_height:height-footer_height, 0:width]
    
    return image

# remove figures and tables based on the minimum area
def exclude_figures_tables(image, contours, min_area=100):
    
    # excludes regions where figures and tables may be
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        print(area)
        if area > min_area:
            # set figure region to white to ignore
            image[y:y+h, x:x+w] = 255
    
    return image


def preprocess_image(image):
    
    # remove the header and footer
    preprocessed_image = remove_header_footer(image)
    
    # deskew the preprocessed image
    deskewed_image = deskew(preprocessed_image)
    
    gray = cv2.cvtColor(deskewed_image, cv2.COLOR_BGR2GRAY)
    
    # threshold the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # exclude figures and tables
    preprocessed_image = exclude_figures_tables(deskewed_image, contours)
    
    return preprocessed_image
    

def get_text(title):
    path = f"papers/{title}"
    flag = True
    extracted_text = ""
    
    pdf_file = path
    
    try: 
        pages = convert_from_path(pdf_file)
    
        for page in pages:
            # preprocess the image 
            preprocessed_image = preprocess_image(page)

            # extract text using OCR
            text = extract_text_from_image(preprocessed_image)
            extracted_text += text
            
    except Exception as e:
        print("Error extracting text")
        flag = False
        
    if flag:
        cleaned_text = clean_text(extracted_text)
        return cleaned_text
    else:
        return None

# add text to the table
def store_text(doi, text):
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    UPDATE scopus_database_v2
                    SET text = %s
                    WHERE doi = %s;
                    """, (text, doi))
        
        conn.commit()
        
    except Exception as e:
        print("Error updating table: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")

# iterate through table doi's
def get_store_text():
    df = fetch_db()
    doi_list = df["doi"]
    title_list = df["title"]
    
    for doi, title in zip(doi_list, title_list):
        pattern = re.compile('[^a-zA-Z ]')
        cleaned_title = pattern.sub('', title)
        filename = cleaned_title + ".pdf"
        text = get_text(filename)
        if text:
            store_text(doi, text)
        
        
