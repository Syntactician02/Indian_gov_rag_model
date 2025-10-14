from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Storage for documents and embeddings
VECTOR_DB_PATH = 'vector_db.faiss'
DOCS_PATH = 'documents.pkl'

class RAGSystem:
    def __init__(self):
        self.documents = []
        self.index = None
        self.load_or_create_db()
    
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
        """Initialize with sample government schemes data"""
        sample_schemes = [
            {
                "title": "Pradhan Mantri Jan Dhan Yojana (PMJDY)",
                "content": "Financial inclusion program launched in 2014. Provides zero-balance bank accounts, RuPay debit cards, accident insurance coverage of ₹1 lakh, and life insurance cover of ₹30,000. Aimed at providing banking facilities to every household.",
                "category": "Financial Inclusion",
                "url": "https://pmjdy.gov.in"
            },
            {
                "title": "Ayushman Bharat - Pradhan Mantri Jan Arogya Yojana (PM-JAY)",
                "content": "World's largest health insurance scheme launched in 2018. Provides health cover of ₹5 lakh per family per year for secondary and tertiary care hospitalization. Covers over 50 crore beneficiaries from poor and vulnerable families.",
                "category": "Healthcare",
                "url": "https://pmjay.gov.in"
            },
            {
                "title": "Pradhan Mantri Awas Yojana (PMAY)",
                "content": "Housing for All scheme with two components: Urban (PMAY-U) and Gramin (PMAY-G). Provides affordable housing to economically weaker sections and low-income groups. Offers interest subsidy on home loans and financial assistance for house construction.",
                "category": "Housing",
                "url": "https://pmaymis.gov.in"
            },
            {
                "title": "Swachh Bharat Mission",
                "content": "Cleanliness campaign launched in 2014 aimed at achieving Open Defecation Free (ODF) India. Focuses on construction of toilets, solid waste management, and behavioral change regarding sanitation practices. Covers both urban and rural areas.",
                "category": "Sanitation",
                "url": "https://swachhbharatmission.gov.in"
            },
            {
                "title": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
                "content": "Income support scheme for farmers launched in 2019. Provides direct income support of ₹6,000 per year to all farmer families in three equal installments of ₹2,000 each. Aims to supplement financial needs of farmers for agricultural inputs.",
                "category": "Agriculture",
                "url": "https://pmkisan.gov.in"
            },
            {
                "title": "Make in India",
                "content": "Initiative launched in 2014 to encourage companies to manufacture products in India. Focuses on 25 sectors including automobiles, chemicals, IT, pharmaceuticals, textiles, and renewable energy. Aims to boost economic growth and generate employment.",
                "category": "Economic Development",
                "url": "https://makeinindia.com"
            },
            {
                "title": "Digital India",
                "content": "Campaign launched in 2015 to transform India into a digitally empowered society. Focuses on digital infrastructure, governance & services on demand, and digital literacy. Includes initiatives like BharatNet, digital payments, e-governance, and Common Service Centers.",
                "category": "Technology",
                "url": "https://digitalindia.gov.in"
            },
            {
                "title": "Beti Bachao Beti Padhao",
                "content": "Social campaign launched in 2015 to address declining Child Sex Ratio and women empowerment. Focuses on prevention of gender-biased sex selection, ensuring survival and protection of girl child, and ensuring education and participation of girls.",
                "category": "Women Empowerment",
                "url": "https://wcd.nic.in"
            },
            {
                "title": "Pradhan Mantri Mudra Yojana (PMMY)",
                "content": "Scheme launched in 2015 to provide loans up to ₹10 lakh to non-corporate, non-farm small/micro enterprises. Three categories: Shishu (up to ₹50,000), Kishore (₹50,001 to ₹5 lakh), and Tarun (₹5 lakh to ₹10 lakh). Promotes entrepreneurship.",
                "category": "Financial Support",
                "url": "https://mudra.org.in"
            },
            {
                "title": "Skill India Mission",
                "content": "Initiative launched in 2015 to train over 40 crore people in different skills by 2022. Includes programs like Pradhan Mantri Kaushal Vikas Yojana (PMKVY), National Apprenticeship Promotion Scheme, and recognition of prior learning.",
                "category": "Skill Development",
                "url": "https://skillindia.gov.in"
            }
        ]
        
        for scheme in sample_schemes:
            self.documents.append({
                "title": scheme["title"],
                "content": scheme["content"],
                "category": scheme["category"],
                "url": scheme["url"],
                "timestamp": datetime.now().isoformat()
            })
        
        self.build_index()
        print(f"Initialized with {len(self.documents)} sample schemes")
    
    def scrape_government_website(self, url):
        """Scrape content from government websites"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text from paragraphs
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            return content[:2000]  # Limit content length
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return None
    
    def add_document(self, title, content, category="General", url=""):
        """Add a new document to the database"""
        doc = {
            "title": title,
            "content": content,
            "category": category,
            "url": url,
            "timestamp": datetime.now().isoformat()
        }
        self.documents.append(doc)
        self.build_index()
        return True
    
    def build_index(self):
        """Build FAISS index from documents"""
        if not self.documents:
            return
        
        texts = [f"{doc['title']} {doc['content']}" for doc in self.documents]
        embeddings = model.encode(texts)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Save to disk
        faiss.write_index(self.index, VECTOR_DB_PATH)
        with open(DOCS_PATH, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def search(self, query, k=3):
        """Search for relevant documents"""
        if not self.index or len(self.documents) == 0:
            return []
        
        query_embedding = model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), min(k, len(self.documents)))
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(1 / (1 + dist))  # Convert distance to similarity
                results.append(doc)
        
        return results

# Initialize RAG system
rag_system = RAGSystem()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat queries"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Search for relevant documents
    results = rag_system.search(query, k=3)
    
    if not results:
        response = "I couldn't find specific information about that. Could you try rephrasing your question or ask about popular schemes like PM-KISAN, Ayushman Bharat, or PMJDY?"
    else:
        # Generate response based on retrieved documents
        response = "Here's what I found about Indian government schemes:\n\n"
        for i, doc in enumerate(results, 1):
            response += f"**{i}. {doc['title']}**\n"
            response += f"{doc['content']}\n"
            response += f"Category: {doc['category']}\n"
            if doc['url']:
                response += f"Learn more: {doc['url']}\n"
            response += "\n"
    
    return jsonify({
        "response": response,
        "sources": results
    })

@app.route('/api/add_scheme', methods=['POST'])
def add_scheme():
    """Add a new scheme to the database"""
    data = request.json
    
    title = data.get('title', '')
    content = data.get('content', '')
    category = data.get('category', 'General')
    url = data.get('url', '')
    
    if not title or not content:
        return jsonify({"error": "Title and content are required"}), 400
    
    rag_system.add_document(title, content, category, url)
    
    return jsonify({"message": "Scheme added successfully", "total_documents": len(rag_system.documents)})

@app.route('/api/scrape', methods=['POST'])
def scrape():
    """Scrape content from a URL and add to database"""
    data = request.json
    url = data.get('url', '')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    content = rag_system.scrape_government_website(url)
    
    if content:
        title = f"Content from {url}"
        rag_system.add_document(title, content, "Scraped", url)
        return jsonify({"message": "Content scraped and added successfully"})
    else:
        return jsonify({"error": "Failed to scrape content"}), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    """Get database statistics"""
    return jsonify({
        "total_documents": len(rag_system.documents),
        "categories": list(set([doc['category'] for doc in rag_system.documents]))
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)







