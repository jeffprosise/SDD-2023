import os, pyodbc, openai
from flask import Flask, render_template, request

app = Flask(__name__)
app.secret_key = os.urandom(24)

conn_string = os.getenv('CONNECTION_STRING')
openai.api_key = os.getenv('OPENAI_KEY')

def answer_query(question):
    content = f'''
        Generate a SQL query to answer the question below from a table with the following schema:
        Products(ProductID, ProductName, SupplierID, CategoryID, QuantityPerUnit, UnitPrice, UnitsInStock, UnitsOnOrder, ReorderLevel, Discontinued)
        Question: {question}
        '''
    
    # Use ChatGPT to generate a SQL query
    messages = [{ 'role': 'user', 'content' : content }]

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=0
    )

    query = response.choices[0].message.content
    
    # Execute the query
    conn = pyodbc.connect(conn_string)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # If the answer contains multiple rows with two or more values each, return a table
    if len(results) > 1 and len(results[0]) > 1:
        content = f'Format the following string representing a Python list into an HTML table: {str(results)}'
        
        messages = [{ 'role': 'user', 'content' : content }]

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            temperature=0
        )
        
        return response.choices[0].message.content

    # If the answer contains multiple rows with one value each, return a multiline list
    elif len(results) > 1:
        return'<br>'.join(result[0] for result in results)    
    
    # Otherwise use ChatGPT to formulate an answer
    else:
        result = results[0][0]
        
        content = f'''
            Phrase an answer to the question below, where the answer to the question is {result}.
            Question: {question}
            '''

        messages = [{ 'role': 'user', 'content' : content }]

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages
        )

        return response.choices[0].message.content

@app.route('/', methods=['GET'])
def index():
    output = ''
    query = request.args.get('query')

    if query is None:
        query = ''

    else:
        output = answer_query(query)

    return render_template('index.html', query=query, answer=output)
