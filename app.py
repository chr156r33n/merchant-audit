import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# Restore stable version from 3 days ago

# Configuration
GEMINI_API_URL = 'https://api.gemini.com/v1/models'

@app.route('/')
def index():
    api_key = request.args.get('api_key')
    models = []
    
    if api_key:
        response = requests.get(GEMINI_API_URL, headers={'Authorization': f'Bearer {api_key}'})
        if response.status_code == 200:
            models = response.json().get('models', [])

    return render_template('index.html', models=models)

if __name__ == '__main__':
    app.run(debug=True)