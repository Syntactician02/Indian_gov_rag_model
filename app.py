from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from datetime import datetime, timedelta
import json
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Storage paths
VECTOR_DB_PATH = 'vector_db.faiss'
DOCS_PATH = 'documents.pkl'
PROJECTS_CACHE_PATH = 'projects_cache.json'
CACHE_EXPIRY_HOURS = 24

class RAGSystem:
    def __init__(self):
        self.documents = []
        self.index = None
        self.projects_cache = {}
        self.load_or_create_db()
        self.load_projects_cache()
    
    def load_projects_cache(self):
        if os.path.exists(PROJECTS_CACHE_PATH):
            try:
                with open(PROJECTS_CACHE_PATH, 'r') as f:
                    cache_data = json.load(f)
                    if cache_data.get('timestamp'):
                        cache_time = datetime.fromisoformat(cache_data['timestamp'])
                        if datetime.now() - cache_time < timedelta(hours=CACHE_EXPIRY_HOURS):
                            self.projects_cache = cache_data.get('projects', {})
                            print(f"Loaded {len(self.projects_cache)} projects from cache")
            except Exception as e:
                print(f"Error loading cache: {e}")
    
    def save_projects_cache(self):
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'projects': self.projects_cache
        }
        with open(PROJECTS_CACHE_PATH, 'w') as f:
            json.dump(cache_data, f)
    
    def load_or_create_db(self):
        if os.path.exists(VECTOR_DB_PATH) and os.path.exists(DOCS_PATH):
            self.index = faiss.read_index(VECTOR_DB_PATH)
            with open(DOCS_PATH, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Loaded {len(self.documents)} documents from database")
        else:
            print("Creating new database")
            self.initialize_with_sample_data()
    
    def initialize_with_sample_data(self):
        sample_data = [
            # Add your sample schemes/projects here
            {
                "title": "Pradhan Mantri Jan Dhan Yojana (PMJDY)",
                "content": "Financial inclusion program launched in 2014. Provides zero-balance bank accounts, RuPay debit cards, accident insurance coverage of ₹1 lakh, and life insurance cover of ₹30,000. Over 50 crore accounts opened. Aimed at providing banking facilities to every household.",
                "category": "Financial Inclusion",
                "url": "https://pmjdy.gov.in",
                "metrics": {"accounts": 500000000, "amount_deposited": 2000000000000}
            },
            {
                "title": "Ayushman Bharat - PM-JAY",
                "content": "World's largest health insurance scheme launched in 2018. Provides health cover of ₹5 lakh per family per year for secondary and tertiary care hospitalization. Covers over 50 crore beneficiaries from poor and vulnerable families. Over 6 crore hospital admissions covered.",
                "category": "Healthcare",
                "url": "https://pmjay.gov.in",
                "metrics": {"beneficiaries": 500000000, "hospitals": 27000}
            }
            # You can add the rest of your original sample_data here
        ]
        
        for item in sample_data:
            self.documents.append({
                "title": item["title"],
                "content": item["content"],
                "category": item["category"],
                "url": item["url"],
                "timestamp": datetime.now().isoformat(),
                "metrics": item.get("metrics", {}),
                "status": item.get("status", "Active")
            })
        
        self.build_index()
        print(f"Initialized with {len(self.documents)} items")
    
    def scrape_government_website(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            return content[:3000]
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None
    
    def add_document(self, title, content, category="General", url="", metrics=None, status="Active"):
        doc = {
            "title": title,
            "content": content,
            "category": category,
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "status": status
        }
        self.documents.append(doc)
        self.build_index()
        return True
    
    def build_index(self):
        if not self.documents:
            return
        texts = [f"{doc['title']} {doc['content']} {doc.get('category', '')}" for doc in self.documents]
        embeddings = model.encode(texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        faiss.write_index(self.index, VECTOR_DB_PATH)
        with open(DOCS_PATH, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def search(self, query, k=5):
        if not self.index or len(self.documents) == 0:
            return []
        query_embedding = model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), min(k, len(self.documents)))
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(1 / (1 + dist))
                results.append(doc)
        return results
    
    def get_category_stats(self):
        stats = defaultdict(int)
        for doc in self.documents:
            stats[doc.get('category', 'General')] += 1
        return dict(stats)
    
    def get_projects_with_metrics(self):
        projects = []
        for doc in self.documents:
            if doc.get('metrics') and doc['metrics']:
                projects.append({
                    'title': doc['title'],
                    'category': doc['category'],
                    'metrics': doc['metrics'],
                    'status': doc.get('status', 'Active')
                })
        return projects

# Initialize RAG system
rag_system = RAGSystem()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    results = rag_system.search(query, k=3)
    results = [doc for doc in results if doc['relevance_score'] > 0.2]
    
    if not results:
        return jsonify({
            "text": "I couldn't find information about that. Try asking about a known scheme or project.",
            "sources": [],
            "chart_data": None
        })
    
    doc = results[0]  # most relevant
    
    response_text = f"**{doc['title']}**\n_{doc['category']}_\n\n"
    response_text += f"{doc['content'][:500]}..."
    
    if doc.get('status'):
        response_text += f"\n📊 Status: {doc['status']}"
    
    if doc.get('metrics'):
        response_text += "\n📈 Key Metrics:\n"
        for key, value in doc['metrics'].items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, (int, float)):
                if value >= 10000000:
                    formatted_value = f"{value/10000000:.1f} Cr"
                elif value >= 100000:
                    formatted_value = f"{value/100000:.1f} Lakh"
                else:
                    formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            response_text += f"  • {formatted_key}: {formatted_value}\n"
    
    if doc.get('url'):
        response_text += f"\n🔗 [Official Website]({doc['url']})"
    
    return jsonify({
        "text": response_text,
        "sources": [doc],
        "chart_data": None
    })

@app.route('/api/add_scheme', methods=['POST'])
def add_scheme():
    data = request.json
    title = data.get('title', '')
    content = data.get('content', '')
    category = data.get('category', 'General')
    url = data.get('url', '')
    metrics = data.get('metrics', {})
    status = data.get('status', 'Active')
    
    if not title or not content:
        return jsonify({"error": "Title and content are required"}), 400
    
    rag_system.add_document(title, content, category, url, metrics, status)
    return jsonify({
        "message": "Added successfully",
        "total_documents": len(rag_system.documents)
    })

@app.route('/api/projects', methods=['GET'])
def get_projects():
    projects = rag_system.get_projects_with_metrics()
    return jsonify({"projects": projects})

@app.route('/api/stats', methods=['GET'])
def stats():
    category_stats = rag_system.get_category_stats()
    return jsonify({
        "total_documents": len(rag_system.documents),
        "categories": list(category_stats.keys()),
        "category_distribution": category_stats,
        "projects_with_metrics": len([d for d in rag_system.documents if d.get('metrics')])
    })

@app.route('/api/categories', methods=['GET'])
def get_categories():
    categories = list(set([doc['category'] for doc in rag_system.documents]))
    return jsonify({"categories": sorted(categories)})

@app.route('/api/search_by_category', methods=['POST'])
def search_by_category():
    data = request.json
    category = data.get('category', '')
    filtered_docs = [doc for doc in rag_system.documents if doc['category'] == category]
    return jsonify({"results": filtered_docs})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
