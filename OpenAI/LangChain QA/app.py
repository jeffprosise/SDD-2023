import os
from flask import Flask, render_template, request
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Assumes OpenAI API key is stored in environment variable named OPENAI_API_KEY

# Add a from_persistent_index method to VectorstoreIndexCreator
def from_persistent_index(self, path: str)-> VectorStoreIndexWrapper:
    vectorstore = self.vectorstore_cls(persist_directory=path, embedding_function=self.embedding)
    return VectorStoreIndexWrapper(vectorstore=vectorstore)

VectorstoreIndexCreator.from_persistent_index=from_persistent_index

# Create an index over an in-memory vector store
db = VectorstoreIndexCreator().from_persistent_index('/Data')

@app.route('/', methods=['GET'])
def index():
    output = ''
    query = request.args.get('query')

    if query is None:
        query = ''

    else:
        output = db.query(query)

    return render_template('index.html', query=query, answer=output)
