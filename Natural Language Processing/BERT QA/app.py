import os, pickle
from flask import Flask, render_template, request
from transformers import TFAutoModel, AutoTokenizer, TFAutoModelForQuestionAnswering, pipeline
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load Hugging Face models
bert_id = 'sebastian-hofstaetter/distilbert-dot-margin_mse-T2-msmarco'
bert_tokenizer = AutoTokenizer.from_pretrained(bert_id) 
bert_model = TFAutoModel.from_pretrained(bert_id, from_pt=True)

qa_id = 'deepset/minilm-uncased-squad2'
qa_tokenizer = AutoTokenizer.from_pretrained(qa_id)
qa_model = TFAutoModelForQuestionAnswering.from_pretrained(qa_id, from_pt=True)
qa_pipe = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

# Load the knowledge base and embedding vectors
contexts = pickle.load(open('contexts.pkl', 'rb'))

# Define functions for processing queries
def vectorize_text(text):
    tokenized_text = bert_tokenizer(text[:512], return_tensors='tf')
    vectorized_text = bert_model(tokenized_text)[0][:, 0, :][0]
    return vectorized_text

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

def order_contexts_by_query_similarity(query, contexts):
    query_embedding = vectorize_text(query).numpy()

    similarities = sorted([
        (vector_similarity(query_embedding, embedding), text) for text, embedding in contexts
    ], reverse=True)

    return similarities

def answer_query(query, max_items=3):
    best_score = 0.0
    best_context = None
    best_start = 0
    best_end = 0

    best_contexts = order_contexts_by_query_similarity(query, contexts)[:max_items]    

    for similarity, text in best_contexts:
        result = qa_pipe(question=query, context=text, handle_impossible_answer=True)

        if result['start'] != result['end']:
            score = result['score']

            if score > best_score:
                best_score = score
                best_context = text
                best_start = result['start']
                best_end = result['end']

    if best_score > 0.0:
        return f'{best_context[:best_start]}<mark>{best_context[best_start:best_end]}' \
               f'</mark>{best_context[best_end:]} ({best_score:.1%})'
    else:
        return 'I don\'t know'
    
@app.route('/', methods=['GET'])
def index():
    output = ''
    query = request.args.get('query')

    if query is None:
        query = ''

    else:
        output = answer_query(query)

    return render_template('index.html', query=query, answer=output)
