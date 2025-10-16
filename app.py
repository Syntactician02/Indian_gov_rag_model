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
import re
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
        """Load cached project data"""
        if os.path.exists(PROJECTS_CACHE_PATH):
            try:
                with open(PROJECTS_CACHE_PATH, 'r') as f:
                    cache_data = json.load(f)
                    # Check if cache is still valid
                    if cache_data.get('timestamp'):
                        cache_time = datetime.fromisoformat(cache_data['timestamp'])
                        if datetime.now() - cache_time < timedelta(hours=CACHE_EXPIRY_HOURS):
                            self.projects_cache = cache_data.get('projects', {})
                            print(f"Loaded {len(self.projects_cache)} projects from cache")
            except Exception as e:
                print(f"Error loading cache: {e}")
    
    def save_projects_cache(self):
        """Save project data to cache"""
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'projects': self.projects_cache
        }
        with open(PROJECTS_CACHE_PATH, 'w') as f:
            json.dump(cache_data, f)
    
    def load_or_create_db(self):
        """Load existing database or create new one"""
        if os.path.exists(VECTOR_DB_PATH) and os.path.exists(DOCS_PATH):
            self.index = faiss.read_index(VECTOR_DB_PATH)
            with open(DOCS_PATH, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Loaded {len(self.documents)} documents from database")
        else:
            print("Creating new database")
            self.initialize_with_sample_data()
    
    def initialize_with_sample_data(self):
        """Initialize with comprehensive government schemes and projects data"""
        sample_data = [
            # Existing schemes
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
            },
            # Infrastructure Projects
            {
                "title": "Delhi Metro Rail Project",
                "content": "India's largest metro network spanning 393 km with 288 stations across Delhi-NCR. Phase 4 expansion of 104 km under construction. Daily ridership of 60+ lakh passengers. Reduces 3.3 million tons of CO2 annually. Connects major areas including Noida, Gurgaon, Faridabad, and Ghaziabad.",
                "category": "Infrastructure",
                "url": "https://delhimetrorail.com",
                "metrics": {"length_km": 393, "stations": 288, "daily_ridership": 6000000, "phase4_length": 104},
                "status": "Ongoing - Phase 4"
            },
            {
                "title": "Mumbai Coastal Road Project",
                "content": "29.2 km freeway project connecting South Mumbai to the Western Suburbs. 8-lane elevated and tunnel sections. Expected to reduce travel time by 70%. Budget of ₹12,700 crore. First phase (10.58 km) operational from May 2024. Will reduce traffic congestion significantly.",
                "category": "Infrastructure",
                "url": "https://mcgm.gov.in",
                "metrics": {"length_km": 29.2, "budget_crore": 12700, "completion": 45},
                "status": "Ongoing - 45% Complete"
            },
            {
                "title": "Bharatmala Pariyojana",
                "content": "Umbrella program for highway development covering 34,800 km. Focuses on border roads, coastal roads, and expressways. Budget of ₹5,35,000 crore. Includes Delhi-Mumbai Expressway (1,386 km) and Chennai-Bangalore Expressway. Will improve logistics efficiency.",
                "category": "Infrastructure",
                "url": "https://bharatmala.in",
                "metrics": {"total_km": 34800, "budget_crore": 535000, "completion": 30},
                "status": "Ongoing"
            },
            {
                "title": "Sagarmala Project",
                "content": "Port-led development program to modernize 200+ Indian ports. Investment of ₹6 lakh crore. Covers port modernization, connectivity enhancement, port-linked industrialization. Will increase port capacity from 1,500 MTPA to 3,000 MTPA by 2025.",
                "category": "Infrastructure",
                "url": "https://sagarmala.gov.in",
                "metrics": {"ports": 200, "budget_crore": 600000, "capacity_mtpa": 3000},
                "status": "Ongoing"
            },
            {
                "title": "PM GatiShakti National Master Plan",
                "content": "₹100 lakh crore infrastructure program launched in 2021. Integrated planning for roads, railways, airports, ports, mass transport, waterways, and logistics. Digital platform for 16 ministries. Aims to reduce logistics costs and improve efficiency.",
                "category": "Infrastructure",
                "url": "https://gatishakti.gov.in",
                "metrics": {"budget_crore": 10000000, "ministries": 16},
                "status": "Ongoing"
            },
            {
                "title": "High-Speed Rail (Bullet Train) Project",
                "content": "Mumbai-Ahmedabad High-Speed Rail Corridor covering 508 km. Speed of 320 kmph. 12 stations including Mumbai, Thane, Surat, Vadodara, Ahmedabad. Travel time reduced from 7 hours to 2 hours. Budget of ₹1,08,000 crore. Technology collaboration with Japan.",
                "category": "Infrastructure",
                "url": "https://nhsrcl.in",
                "metrics": {"length_km": 508, "speed_kmph": 320, "stations": 12, "budget_crore": 108000, "completion": 35},
                "status": "Ongoing - 35% Complete"
            },
            # Smart Cities
            {
                "title": "Smart Cities Mission",
                "content": "Urban transformation program covering 100 cities. Focus on smart infrastructure, sustainable environment, and improved quality of life. Investment of ₹2.1 lakh crore. Over 8,000 projects completed including smart roads, smart parking, integrated command centers.",
                "category": "Urban Development",
                "url": "https://smartcities.gov.in",
                "metrics": {"cities": 100, "projects": 8000, "budget_crore": 210000},
                "status": "Ongoing"
            },
            # More schemes
            {
                "title": "PM-KISAN",
                "content": "Income support scheme for farmers launched in 2019. ₹6,000 per year to 11 crore farmer families in three installments. Direct benefit transfer to bank accounts. Over ₹2.8 lakh crore disbursed so far.",
                "category": "Agriculture",
                "url": "https://pmkisan.gov.in",
                "metrics": {"beneficiaries": 110000000, "amount_per_year": 6000}
            },
            {
                "title": "National Solar Mission (Jawaharlal Nehru National Solar Mission)",
                "content": "Target of 100 GW solar power capacity by 2022, extended to 280 GW by 2030. Includes rooftop solar, solar parks, and grid-connected systems. Currently over 70 GW installed. Promotes clean energy and reduces carbon footprint.",
                "category": "Renewable Energy",
                "url": "https://mnre.gov.in",
                "metrics": {"target_gw": 280, "current_gw": 70, "completion": 25},
                "status": "Ongoing"
            }
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
    
    def fetch_live_project_data(self, project_name):
        """Fetch live project updates from web"""
        try:
            # Search for recent news about the project
            search_query = f"{project_name} India latest update progress"
            search_url = f"https://www.google.com/search?q={requests.utils.quote(search_query)}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Note: In production, use proper news APIs or government APIs
            # This is a simplified version for demonstration
            
            return {
                "last_updated": datetime.now().isoformat(),
                "source": "web_search",
                "status": "Data fetched successfully"
            }
            
        except Exception as e:
            print(f"Error fetching live data: {e}")
            return None
    
    def scrape_government_website(self, url):
        """Scrape content from government websites"""
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
        """Add a new document to the database"""
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
        """Build FAISS index from documents"""
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
        """Search for relevant documents with enhanced ranking"""
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
        """Get statistics by category"""
        stats = defaultdict(int)
        for doc in self.documents:
            stats[doc.get('category', 'General')] += 1
        return dict(stats)
    
    def get_projects_with_metrics(self):
        """Get all projects that have metrics data"""
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Enhanced chat with better responses and chart suggestions"""
    data = request.json
    query = data.get('query', '').lower()
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Search for relevant documents
    results = rag_system.search(query, k=5)
    
    # Check if query is about comparisons or stats
    show_chart = any(keyword in query for keyword in ['compare', 'statistics', 'stats', 'graph', 'chart', 'data', 'progress', 'metrics'])
    
    if not results:
        response = {
            "text": "I couldn't find specific information about that. Try asking about:\n• Delhi Metro Project\n• Mumbai Coastal Road\n• Bullet Train Project\n• Smart Cities Mission\n• PM-KISAN\n• Ayushman Bharat",
            "sources": [],
            "chart_data": None
        }
    else:
        # Generate comprehensive response
        response_text = "📋 **Here's what I found:**\n\n"
        chart_data = None
        
        for i, doc in enumerate(results[:3], 1):
            response_text += f"**{i}. {doc['title']}**\n"
            response_text += f"_{doc['category']}_\n\n"
            response_text += f"{doc['content'][:300]}...\n\n"
            
            if doc.get('status'):
                response_text += f"📊 Status: {doc['status']}\n"
            
            if doc.get('metrics'):
                response_text += "📈 Key Metrics:\n"
                for key, value in list(doc['metrics'].items())[:4]:
                    formatted_key = key.replace('_', ' ').title()
                    if isinstance(value, (int, float)) and value > 1000000:
                        formatted_value = f"{value/10000000:.1f} Crore" if value > 10000000 else f"{value/100000:.1f} Lakh"
                    else:
                        formatted_value = f"{value:,}" if isinstance(value, (int, float)) else value
                    response_text += f"  • {formatted_key}: {formatted_value}\n"
                response_text += "\n"
            
            if doc['url']:
                response_text += f"🔗 [Official Website]({doc['url']})\n"
            response_text += "\n---\n\n"
        
        # Prepare chart data if metrics are available
        if show_chart and any(doc.get('metrics') for doc in results[:3]):
            chart_data = {
                "type": "bar",
                "labels": [doc['title'][:30] for doc in results[:3] if doc.get('metrics')],
                "datasets": []
            }
            
            # Find common metrics across projects
            all_metrics = {}
            for doc in results[:3]:
                if doc.get('metrics'):
                    for key, value in doc['metrics'].items():
                        if isinstance(value, (int, float)):
                            if key not in all_metrics:
                                all_metrics[key] = []
                            all_metrics[key].append(value)
            
            # Create dataset for the most common metric
            if all_metrics:
                metric_key = max(all_metrics.keys(), key=lambda k: len(all_metrics[k]))
                chart_data["datasets"].append({
                    "label": metric_key.replace('_', ' ').title(),
                    "data": all_metrics[metric_key],
                    "backgroundColor": ['#3b82f6', '#10b981', '#f59e0b'][:len(all_metrics[metric_key])]
                })
        
        response = {
            "text": response_text,
            "sources": results[:3],
            "chart_data": chart_data
        }
    
    return jsonify(response)

@app.route('/api/add_scheme', methods=['POST'])
def add_scheme():
    """Add a new scheme/project to the database"""
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
    """Get all projects with metrics"""
    projects = rag_system.get_projects_with_metrics()
    return jsonify({"projects": projects})

@app.route('/api/stats', methods=['GET'])
def stats():
    """Get comprehensive database statistics"""
    category_stats = rag_system.get_category_stats()
    
    return jsonify({
        "total_documents": len(rag_system.documents),
        "categories": list(category_stats.keys()),
        "category_distribution": category_stats,
        "projects_with_metrics": len([d for d in rag_system.documents if d.get('metrics')])
    })

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all available categories"""
    categories = list(set([doc['category'] for doc in rag_system.documents]))
    return jsonify({"categories": sorted(categories)})

@app.route('/api/search_by_category', methods=['POST'])
def search_by_category():
    """Search projects by category"""
    data = request.json
    category = data.get('category', '')
    
    filtered_docs = [doc for doc in rag_system.documents if doc['category'] == category]
    return jsonify({"results": filtered_docs})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
