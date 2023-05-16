import os, pickle
from flask import Flask, render_template, request
from transformers import TFAutoModel, AutoTokenizer, TFAutoModelForQuestionAnswering, pipeline
import numpy as np

# Create Flask app and set secret key
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize BERT and Question Answering models
bert_id = 'sebastian-hofstaetter/distilbert-dot-margin_mse-T2-msmarco'
bert_tokenizer = AutoTokenizer.from_pretrained(bert_id) 
bert_model = TFAutoModel.from_pretrained(bert_id, from_pt=True)

qa_id = 'deepset/minilm-uncased-squad2'
qa_tokenizer = AutoTokenizer.from_pretrained(qa_id)
qa_model = TFAutoModelForQuestionAnswering.from_pretrained(qa_id, from_pt=True)
qa_pipe = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

# Load preprocessed contexts
contexts = pickle.load(open('contexts.pkl', 'rb'))

# Helper function to convert text to a vector
def vectorize_text(text):
    tokenized_text = bert_tokenizer(text[:512], return_tensors='tf') # tokenize text
    vectorized_text = bert_model(tokenized_text)[0][:, 0, :][0] # get BERT encoding
    return vectorized_text

# Helper function to calculate cosine similarity between two vectors
def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

# Helper function to order contexts by similarity to query
def order_contexts_by_query_similarity(query, contexts):
    query_embedding = vectorize_text(query).numpy() # convert query to BERT vector

    similarities = sorted([
        (vector_similarity(query_embedding, embedding), text) for text, embedding in contexts # calculate cosine similarity for each context
    ], reverse=True) # sort contexts by descending similarity 

    return similarities

# Function to answer a given query
def answer_query(query, max_items=3):
    best_score = 0.0
    best_context = None
    best_start = 0
    best_end = 0

    best_contexts = order_contexts_by_query_similarity(query, contexts)[:max_items] # get top contexts based on similarity to query

    # Loop through top contexts and use QA model to find answer
    for similarity, text in best_contexts:
        result = qa_pipe(question=query, context=text, handle_impossible_answer=True)

        # If answer is found, update best answer
        if result['start'] != result['end']:
            score = result['score']

            if score > best_score:
                best_score = score
                best_context = text
                best_start = result['start']
                best_end = result['end']

    # If answer is found, format it with HTML tags and return
    if best_score > 0.0:
        return f'{best_context[:best_start]}<mark>{best_context[best_start:best_end]}' \
               f'</mark>{best_context[best_end:]} ({best_score:.1%})'
    # Otherwise, return "I don't know"
    else:
        return 'I don\'t know'
    
# Define route for index page
@app.route('/', methods=['GET'])
def index():
    output = ''
    query = request.args.get('query') # get query string from URL parameter

    # If no query is given, set query to empty string
    if query is None:
        query = ''

    # Otherwise, try to answer the query
    else:
        output = answer_query(query)

    # Render index HTML template with query and answer
    return render_template('index.html', query=query, answer=output)