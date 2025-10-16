# 🚀 Complete Setup Guide for GovSchemes AI

## For Hackathon Judges & First-Time Users

This guide will help you set up and run the project in under 5 minutes!

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Detailed Setup](#detailed-setup)
3. [Testing the Application](#testing)
4. [Demo Script](#demo-script)
5. [Troubleshooting](#troubleshooting)

---

## ⚡ Quick Start (2 Minutes)

```bash
# 1. Navigate to project directory
cd govschemes-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py

# 4. Open browser and go to:
# http://localhost:5000
```

**That's it!** The app will automatically:
- Download the ML model (first time only, ~90MB)
- Initialize the database with sample schemes
- Start the web server

---

## 📚 Detailed Setup

### Step 1: System Requirements

**Minimum Requirements:**
- Python 3.8+
- 4GB RAM
- 500MB free disk space
- Internet connection (for first-time model download)

**Recommended:**
- Python 3.9+
- 8GB RAM
- SSD storage

### Step 2: Python Environment

**Check Python version:**
```bash
python --version
# Should show Python 3.8.0 or higher
```

**Create virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**Verify activation:**
You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Expected installation time:** 2-5 minutes

**Common packages installed:**
- Flask (web framework)
- Sentence Transformers (AI embeddings)
- FAISS (vector search)
- BeautifulSoup (web scraping)
- Chart.js dependencies

### Step 4: First Run

```bash
python app.py
```

**What happens on first run:**
1. Downloads ML model (~90MB) - This happens only once
2. Creates vector database files
3. Initializes with 10+ sample schemes
4. Starts Flask server on port 5000

**Expected output:**
```
Creating new database
Initialized with 11 items
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

---

## 🧪 Testing the Application

### Basic Functionality Test

1. **Open browser:** `http://localhost:5000`

2. **Try these queries:**
   ```
   Query 1: "Tell me about Delhi Metro"
   Expected: Information about Delhi Metro with stats
   
   Query 2: "Show infrastructure projects"
   Expected: List of infrastructure projects
   
   Query 3: "Compare projects with statistics"
   Expected: Response with chart visualization
   ```

3. **Check sidebar:**
   - Should show total documents (11+)
   - Should show categories
   - Should show active projects

### API Testing

**Test with curl or Postman:**

```bash
# Test chat endpoint
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Delhi Metro"}'

# Test stats endpoint
curl http://localhost:5000/api/stats

# Test categories endpoint
curl http://localhost:5000/api/categories
```

---

## 🎭 Demo Script for Hackathon Presentation

### 1. Introduction (30 seconds)
"Hi! I'm presenting GovSchemes AI - an intelligent chatbot that helps citizens discover and understand Indian government schemes and infrastructure projects using AI."

### 2. Problem Statement (30 seconds)
"Many citizens are unaware of government schemes they're eligible for. Information is scattered across multiple websites, making it hard to find relevant schemes quickly."

### 3. Solution Demo (2 minutes)

**Demo Flow:**

1. **Basic Query:**
   - Type: "What is PM-KISAN?"
   - Highlight: Instant semantic search, clear response with official links

2. **Infrastructure Project:**
   - Type: "Tell me about Delhi Metro project progress"
   - Highlight: Real-time project data with metrics (stations, ridership)

3. **Comparison with Charts:**
   - Type: "Compare infrastructure projects statistics"
   - Highlight: Interactive chart visualization

4. **Category Browse:**
   - Click on sidebar categories
   - Show: How users can explore by category
