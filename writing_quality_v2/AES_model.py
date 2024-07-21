from transformers import AutoModelForSequenceClassification, AutoTokenizer
import psycopg2
from data.config import DB_NAME, DB_HOST, DB_USER, DB_PASS, DB_PORT
import torch
import numpy as np
import clean_text
import readability_extraction as readability
import data.scopus_data_retrieval_v4 as scopus

model = AutoModelForSequenceClassification.from_pretrained("rong4ivy/EnglishEssay_Scoring_LM")
tokenizer = AutoTokenizer.from_pretrained("rong4ivy/EnglishEssay_Scoring_LM")

# splitting the tokens
def split_overlap(tensor, chunk_size, stride, min_chunk_len):
    result = [tensor[i : i + chunk_size] for i in range(0, len(tensor), stride)]
    if len(result) > 1:
        result = [x for x in result if len(x) >= min_chunk_len]
    return result 

# adding special tokens to the beginning and end
def add_special_tokens(input_id_chunks, mask_chunks):
    for i in range(len(input_id_chunks)):
        input_id_chunks[i] = torch.cat([torch.Tensor([101]), input_id_chunks[i], torch.Tensor([102])])
        mask_chunks[i] = torch.cat([torch.Tensor([1]), mask_chunks[i], torch.Tensor([1])])

# add padding tokens so all chunks have size 512
def add_padding(input_id_chunks, mask_chunks):
    for i in range(len(input_id_chunks)):
        pad_len = 512 - input_id_chunks[i].shape[0]
        if pad_len > 0:
            input_id_chunks[i] = torch.cat([input_id_chunks[i], torch.Tensor([0] * pad_len)])
            mask_chunks[i] = torch.cat([mask_chunks[i], torch.Tensor([0] * pad_len)])

# stacking the tensors
def stack_tokens(input_id_chunks, mask_chunks):
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)
    return input_ids.long(), attention_mask.int()

# transforming text to be able to be processed by BERT-based model
def transform_text(text, tokenizer, chunk_size, stride, min_chunk_len):
    encoded_input = tokenizer(text, return_tensors='pt')
    input_id_chunks = split_overlap(encoded_input["input_ids"][0], chunk_size, stride, min_chunk_len)
    mask_chunks = split_overlap(encoded_input["attention_mask"][0], chunk_size, stride, min_chunk_len)
    add_special_tokens(input_id_chunks, mask_chunks)
    add_padding(input_id_chunks, mask_chunks)
    input_ids, attention_mask = stack_tokens(input_id_chunks, mask_chunks)
    return input_ids, attention_mask 
    
# updating table with writing quality metrics
def add_quality_metrics(cohesion, syntax, vocabulary, phraseology, grammar, conventions, doi):
    try:
        conn = psycopg2.connect(database=DB_NAME,
                        host=DB_HOST,
                        user=DB_USER,
                        password=DB_PASS,
                        port=DB_PORT)
        
        cur = conn.cursor()
        
        cur.execute("""
                    UPDATE scopus_database_v4
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
        print("Error updating table: ", e)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print("Database connection closed")

# calculating the writing quality and finetuning the model to accept longer texts   
def compute_quality(text, doi):
    model.eval()
    input_ids, attention_mask = transform_text(text, tokenizer, 510, 510, 1)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    # get scores from the model
    predictions = outputs.logits.squeeze()
    predicted_scores = predictions.numpy()
    scores = 2.25 * predicted_scores - 1.25
    scores = np.array(scores)
    if scores.ndim > 1:
        scores = np.mean(scores, axis=0)
    
    add_quality_metrics(float(scores[0]), float(scores[1]), float(scores[2]), float(scores[3]), float(scores[4]), float(scores[5]), doi)
    
# compute the writing quality and readability metrics of each text in the database
def quality():
    df = scopus.fetch_db()
    doi_list = df["doi"]
    text_list = df["text"]
    
    for doi, text in zip(doi_list, text_list):
        text = clean_text.replace_unnecessary_chars(text)
        compute_quality(text, doi)
        readability.compute_readability_metrics(text, doi)
        
