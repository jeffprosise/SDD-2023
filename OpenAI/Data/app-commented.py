# import necessary packages
import os, pickle
from flask import Flask, render_template, request
from transformers import TFAutoModel, AutoTokenizer, TFAutoModelForQuestionAnswering, pipeline
import numpy as np

# initialize Flask app and secret key
app = Flask(__name__)
app.secret_key = os.urandom(24)

# define the BERT-based language model and tokenizer
bert_id = 'sebastian-hofstaetter/distilbert-dot-margin_mse-T2-msmarco'
bert_tokenizer = AutoTokenizer.from_pretrained(bert_id) 
bert_model = TFAutoModel.from_pretrained(bert_id, from_pt=True)

# define the question-answering model and tokenizer
qa_id = 'deepset/minilm-uncased-squad2'
qa_tokenizer = AutoTokenizer.from_pretrained(qa_id)
qa_model = TFAutoModelForQuestionAnswering.from_pretrained(qa_id, from_pt=True)
qa_pipe = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

# load preprocessed contexts
contexts = pickle.load(open('contexts.pkl', 'rb'))

# define a function for vectorizing text
def vectorize_text(text):
    tokenized_text = bert_tokenizer(text[:512], return_tensors='tf') # tokenize the text input, truncating to 512 tokens or less
    vectorized_text = bert_model(tokenized_text)[0][:, 0, :][0] # feed the tokenized text into the language model and extract the first (CLS) token's output vector as a representation of the text
    return vectorized_text

# define a function for calculating cosine similarity between two vectors
def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

# define a function for ranking contexts based on similarity to a given query
def order_contexts_by_query_similarity(query, contexts):
    query_embedding = vectorize_text(query).numpy() # vectorize the query using the defined function

    similarities = sorted([
        (vector_similarity(query_embedding, embedding), text) for text, embedding in contexts # calculate cosine similarity between the query embedding and each context
    ], reverse=True) # sort the similarities in descending order

    return similarities

# define a function for answering a given query
def answer_query(query, max_items=3):
    best_score = 0.0
    best_context = None
    best_start = 0
    best_end = 0

    best_contexts = order_contexts_by_query_similarity(query, contexts)[:max_items] # rank the contexts using the defined function and keep only the top max_items    

    for similarity, text in best_contexts:
        result = qa_pipe(question=query, context=text, handle_impossible_answer=True) # use the question answering pipeline to find the answer to the query within the context

        if result['start'] != result['end']: # check that an answer was found
            score = result['score']

            if score > best_score: # update the best answer if a higher score is found
                best_score = score
                best_context = text
                best_start = result['start']
                best_end = result['end']

    if best_score > 0.0: # if a non-zero best score was found, return the answer with highlighted text
        return f'{best_context[:best_start]}<mark>{best_context[best_start:best_end]}' \
               f'</mark>{best_context[best_end:]} ({best_score:.1%})'
    else: # otherwise, return 'I don't know'
        return 'I don\'t know'
    
# define a Flask route for the home page
@app.route('/', methods=['GET'])
def index():
    output = ''
    query = request.args.get('query')

    if query is None:
        query = ''

    else:
        output = answer_query(query)

    return render_template('index.html', query=query, answer=output) # render the HTML template with the query and answer variables