import os, pickle, openai
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Get an OpenAI API key
openai.api_key = os.environ['API_KEY']

# Load the knowledge base and embedding vectors
contexts = pickle.load(open('contexts.pkl', 'rb'))

# Define functions for processing queries
def vectorize_text(text):
    result = openai.Embedding.create(model='text-embedding-ada-002', input=text)
    return result.data[0].embedding

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

def order_contexts_by_query_similarity(query, contexts):
    query_embedding = vectorize_text(query)

    similarities = sorted([
        (vector_similarity(query_embedding, embedding), text) for text, embedding in contexts
    ], reverse=True)

    return similarities

def answer_query(query, max_items=5):
    similar_contexts = order_contexts_by_query_similarity(query, contexts)[:max_items]
    context = ' '.join(text for (similarity, text) in similar_contexts)
    
    content = f'Answer the following question using the provided context, and if the ' \
              f'answer is not contained within the context, say "I don\'t know."\n\n' \
              f'Context: {context}\n\n' \
              f'Q: {query}\n\n' \
              f'A: '

    messages = [
        { 'role': 'user', 'content': content }
    ]

    result = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages
    )
    
    return result.choices[0].message.content

@app.route('/', methods=['GET'])
def index():
    output = ''
    query = request.args.get('query')

    if query is None:
        query = ''

    else:
        output = answer_query(query)

    return render_template('index.html', query=query, answer=output)
