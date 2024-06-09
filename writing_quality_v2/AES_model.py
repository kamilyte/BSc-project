import sys
sys.path.append('/Users/kamile/Desktop/Bachelor-Project/BSc-project')
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import psycopg2
from data.config import DB_NAME, DB_HOST, DB_USER, DB_PASS, DB_PORT, API_KEY, SCOPUS_API_KEY, SCOPUS_BASE_URL, SCOPUS_TOKEN, PLUMX_BASE_URL
import torch
import numpy as np
from paper import text
import clean_text

model = AutoModelForSequenceClassification.from_pretrained("rong4ivy/EnglishEssay_Scoring_LM")
tokenizer = AutoTokenizer.from_pretrained("rong4ivy/EnglishEssay_Scoring_LM")

def split_overlap(tensor, chunk_size, stride, min_chunk_len):
    result = [tensor[i : i + chunk_size] for i in range(0, len(tensor), stride)]
    if len(result) > 1:
        result = [x for x in result if len(x) >= min_chunk_len]
    return result 

def add_special_tokens(input_id_chunks, mask_chunks):
    for i in range(len(input_id_chunks)):
        input_id_chunks[i] = torch.cat([torch.Tensor([101]), input_id_chunks[i], torch.Tensor([102])])
        mask_chunks[i] = torch.cat([torch.Tensor([1]), mask_chunks[i], torch.Tensor([1])])
        
def add_padding(input_id_chunks, mask_chunks):
    for i in range(len(input_id_chunks)):
        pad_len = 512 - input_id_chunks[i].shape[0]
        if pad_len > 0:
            input_id_chunks[i] = torch.cat([input_id_chunks[i], torch.Tensor([0] * pad_len)])
            mask_chunks[i] = torch.cat([mask_chunks[i], torch.Tensor([0] * pad_len)])
            
def stack_tokens(input_id_chunks, mask_chunks):
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)
    return input_ids.long(), attention_mask.int()


def transform_text(text, tokenizer, chunk_size, stride, min_chunk_len):
    encoded_input = tokenizer(text, return_tensors='pt')
    input_id_chunks = split_overlap(encoded_input["input_ids"][0], chunk_size, stride, min_chunk_len)
    mask_chunks = split_overlap(encoded_input["attention_mask"][0], chunk_size, stride, min_chunk_len)
    add_special_tokens(input_id_chunks, mask_chunks)
    add_padding(input_id_chunks, mask_chunks)
    input_ids, attention_mask = stack_tokens(input_id_chunks, mask_chunks)
    return input_ids, attention_mask 
    

# encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=False, truncation=False)
# model.eval()


# input_ids, attention_mask = transform_text(text, tokenizer, 510, 510, 1)
# print(input_ids.shape)

# with torch.no_grad():
#     outputs = model(input_ids, attention_mask)
    
# predictions = outputs.logits.squeeze()
# predicted_scores = predictions.numpy()  
# item_names = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

# scaled_scores = 1 + 4 * (predicted_scores - np.min(predicted_scores)) / (np.max(predicted_scores) - np.min(predicted_scores))
    
# rounded_scores = np.round(scaled_scores * 2) / 2
# Scale predictions from 1 to 10 and round to the nearest 0.5
# scaled_scores = 2.25 * predicted_scores - 1.25
# rounded_scores = [np.round(score * 2) / 2 for score in scaled_scores]  # Round to nearest 0.5
# average_scores = np.mean(rounded_scores, axis=0)

# print(average_scores)
# print(len(average_scores))

# print(item_names[0], average_scores[0])

# for item, score in zip(item_names, average_scores):
#     print(f"{item}: {score: .1f}")
    
def add_quality_metrics(cohesion, syntax, vocabulary, phraseology, grammar, conventions, doi):
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    UPDATE scopus_database
                    SET cohesion = %s,
                        syntax = %s,
                        vocabulary = %s,
                        phraseology = %s,
                        grammar = %s,
                        conventions = %s
                    WHERE doi = %s;
                    """, (cohesion, syntax, vocabulary, phraseology, grammar, conventions, doi))
        
        conn.commit()
        
    except Exception as e:
        print("Error updating: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")
            
            
def compute_quality(text, doi):
    #encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=False, truncation=False)
    model.eval()
    input_ids, attention_mask = transform_text(text, tokenizer, 510, 510, 1)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    predictions = outputs.logits.squeeze()
    predicted_scores = predictions.numpy()  
    item_names = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    
    scaled_scores = 2.25 * predicted_scores - 1.25
    rounded_scores = [np.round(score * 2) / 2 for score in scaled_scores]  # Round to nearest 0.5
    average_scores = np.mean(rounded_scores, axis=0)
    
    add_quality_metrics(float(average_scores[0]), float(average_scores[1]), float(average_scores[2]), float(average_scores[3]), float(average_scores[4]), float(average_scores[5]), doi)
    

def quality():
    df = clean_text.fetch_db()
    doi_list = df["doi"]
    text_list = df["text"]
    
    for doi, text in zip(doi_list, text_list):
        text = clean_text.replace_unnecessary_chars(text)
        compute_quality(text, doi)
        
quality()
print(clean_text.fetch_db())