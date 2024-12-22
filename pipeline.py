# Importing all the required libraries
import pandas as pd
import numpy as np
import torch
import gensim
import gensim.downloader
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Using MPS Device as mac is Apple Silicon, you can change the device type to either "cuda" or "cpu"
device = torch.device('mps')

# Initialzing Bert Tokenizer, Bert Model, Sentence Transformers, Glove and Word2Vec for Vector Embeddings
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
model_bert = AutoModel.from_pretrained("bert-base-uncased").to(device)

model_st = SentenceTransformer("all-mpnet-base-v2", device="mps")

glove_vectors = gensim.downloader.load('glove-twitter-200')

word2vec_vectors = gensim.downloader.load('word2vec-google-news-300')

# OpenAI API Client for OpenAI Embeddings(Just in Case)
client_openai = OpenAI(api_key='your_openai_api_key_here')

## Uncomment the below and comment out the BART for using Llama-3.1-8B for summarization
# Summarization model
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# model_id = "meta-llama/Llama-3.1-8B"
# model_summarizer = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float16)

# Using Facebook's BART for summarization
model_id = "facebook/bart-large-cnn"
model_summarizer = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float16)
model_summarizer = model_summarizer.to(device)

sum_model = pipeline(
    "summarization",
    model=model_summarizer,
    tokenizer=model_id,
    clean_up_tokenization_spaces=True,
    max_length=512,
    truncation=True
)


# Function to get OpenAI embeddings
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client_openai.embeddings.create(input=[text], model=model).data[0].embedding

# BERT Embedding Function
def get_sentence_embedding(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding


# Glove and Word2Vec Sentence Embedding Function
def sentence_embedding(sentence, model):
    word_embeddings = []
    for word in sentence.split():
        if word in model:
            word_embeddings.append(model[word])
    if len(word_embeddings) == 0:
        return np.zeros(model.vector_size)
    else:
        return np.mean(word_embeddings, axis=0)

# Cosine similarity calculation for different embeddings
def cosine_similarity_model(sentence1, sentence2, model):
    embedding1 = sentence_embedding(sentence1, model)
    embedding2 = sentence_embedding(sentence2, model)
    if np.all(embedding1 == 0) or np.all(embedding2 == 0):
        return 0
    return cosine_similarity([embedding1], [embedding2])[0][0]

def cosine_similarity_bert(sentence1, sentence2, tokenizer, model):
    embedding1 = get_sentence_embedding(sentence1, tokenizer, model)
    embedding2 = get_sentence_embedding(sentence2, tokenizer, model)
    return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=1).item()

def cosine_similarity_openai(sentence1, sentence2, model="text-embedding-3-small"):
    embedding1 = [get_embedding(sentence1, model=model)]
    embedding2 = [get_embedding(sentence2, model=model)]
    return cosine_similarity(embedding1, embedding2)[0][0]

def cosine_similarity_st_transformer(issue, summary, st_model):
    embed_1 = [st_model.encode(issue)]
    embed_2 = [st_model.encode(summary)]

    return cosine_similarity(embed_1, embed_2)[0][0]

# Filtering out the Make, Model, Year from dataframe and finding out most relevant rows (by similarities)
# Vector Embedding Options ("bert", "glove", "word2vec", "openai"(put in openai API key) or "sentence_transformer")
def filter_and_calculate_similarities(df, query, embedding_model_type="sentence_transformer"):
    make = query['make']
    model = query['model']
    year = query['year']
    issue = query['issue']

    # Filter the dataframe based on make, model, and year
    filtered_df = df[(df['MAKETXT'].str.lower() == make.lower()) &
                     (df['MODELTXT'].str.lower() == model.lower()) &
                     (df['YEARTXT'].astype(str) == str(year))]
    
    similarities = []
    # Finding Out similarity between each row with the query issue
    for idx, row in filtered_df.iterrows():
        summary = row['SUMMARY']
        if embedding_model_type == "bert":
            similarity = cosine_similarity_bert(issue, summary, tokenizer_bert, model_bert)
        elif embedding_model_type == "glove":
            similarity = cosine_similarity_model(issue, summary, glove_vectors)
        elif embedding_model_type == "word2vec":
            similarity = cosine_similarity_model(issue, summary, word2vec_vectors)
        elif embedding_model_type == "openai":
            similarity = cosine_similarity_openai(issue, summary)
        elif embedding_model_type == "sentence_transformer":
            similarity = cosine_similarity_st_transformer(issue, summary, model_st)
        else:
            raise ValueError("Invalid embedding model type. Choose from 'bert', 'glove', 'word2vec', 'openai', 'sentence_transformer'.")

        similarities.append((idx, similarity))
    # Sorting in Descending Order for most relevant rows
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    unique_summaries = {}
    # Best 10 Unique Rows for Summarization
    for idx, sim in sorted_similarities:
        summary = df.loc[idx, 'SUMMARY']
        if summary not in unique_summaries:
            unique_summaries[summary] = (idx, sim)
        if len(unique_summaries) == 10:
            break
    
    top_indices = [unique_summaries[summary][0] for summary in unique_summaries]
    top_rows = df.loc[top_indices].copy()
    combined_text = " ".join(top_rows['SUMMARY'])
    return top_rows, combined_text

if __name__ == "__main__":
    # Reading CSV File
    df = pd.read_csv('dataset/FLAT_RCL_3.csv')
    # Query Object
    query = {
        'make': 'ford',
        'model': 'escape',
        'year': '2001',
        'issue': 'stuck throttle risk'
    }
    embedding_model_type = "sentence_transformer"
    relevant_rows, combined_text = filter_and_calculate_similarities(df, query, embedding_model_type)
    summary = sum_model(combined_text)
    # Results
    print(f"Top 10 Relevant Rows According to {embedding_model_type}: ")
    print(relevant_rows)
    print(f"\nSummarized Text: {summary[0]['summary_text']}")
 
