# 🇮🇳 GovSchemes AI - Intelligent Government Schemes Chatbot

An advanced RAG (Retrieval-Augmented Generation) chatbot that provides real-time information about Indian government schemes and infrastructure projects with interactive data visualizations.

## 🚀 Features

### Core Features
- ✨ **Semantic Search**: Uses sentence transformers for intelligent query understanding
- 🗄️ **Vector Database**: FAISS-powered fast retrieval system
- 📊 **Data Visualization**: Dynamic charts for project metrics and comparisons
- 🔄 **Real-time Updates**: Web scraping capabilities for latest project information
- 💾 **Persistent Storage**: Automatic caching with pickle and FAISS index
- 🎨 **Modern UI**: Sleek dark-mode interface with smooth animations

### Pre-loaded Information
- **Infrastructure Projects**: Delhi Metro, Mumbai Coastal Road, Bullet Train, Bharatmala, Sagarmala
- **Government Schemes**: PM-KISAN, Ayushman Bharat, PMJDY, Smart Cities Mission
- **Categories**: Healthcare, Infrastructure, Financial Inclusion, Agriculture, Urban Development, and more

## 📸 Screenshots

### Main Chat Interface
![Chat Interface](screenshots/chat.png)

### Data Visualization
![Charts](screenshots/charts.png)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended for better performance)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/govschemes-ai.git
cd govschemes-ai
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

5. **Open in browser**
```
http://localhost:5000
```

## 📁 Project Structure

```
govschemes-ai/
│
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Frontend UI
├── vector_db.faiss       # FAISS vector index (auto-generated)
├── documents.pkl         # Pickled documents (auto-generated)
├── projects_cache.json   # Cached project data (auto-generated)
└── README.md            # This file
```

## 🎯 Usage

### Basic Queries
```
"Tell me about Delhi Metro project"
"What is PM-KISAN scheme?"
"Show me healthcare schemes"
```

### Advanced Queries
```
"Compare infrastructure projects"
"Show statistics for Smart Cities"
"What's the progress of Mumbai Coastal Road?"
```

### Adding New Schemes
Use the API endpoint:
```bash
curl -X POST http://localhost:5000/api/add_scheme \
  -H "Content-Type: application/json" \
  -d '{
    "title": "New Scheme Name",
    "content": "Description of the scheme...",
    "category": "Category Name",
    "url": "https://official-website.gov.in",
    "metrics": {
      "beneficiaries": 1000000,
      "budget_crore": 5000
    }
  }'
```

## 🔧 API Endpoints

### Chat
- **POST** `/api/chat`
  - Body: `{"query": "your question"}`
  - Returns: Response with sources and optional chart data

### Statistics
- **GET** `/api/stats`
  - Returns: Total documents, categories, and project count

### Add Scheme
- **POST** `/api/add_scheme`
  - Body: `{"title": "...", "content": "...", "category": "...", "url": "...", "metrics": {...}}`

### Get Projects
- **GET** `/api/projects`
  - Returns: All projects with metrics

### Categories
- **GET** `/api/categories`
  - Returns: List of all available categories

## 🎨 Customization

### Adding More Data
Edit the `initialize_with_sample_data()` method in `app.py` to add more schemes.

### Changing Model
Replace `'all-MiniLM-L6-v2'` with any sentence-transformers model:
```python
model = SentenceTransformer('your-model-name')
```

### UI Theming
Modify CSS variables in `index.html`:
```css
:root {
    --primary: #2563eb;
    --secondary: #10b981;
    /* Add your colors */
}
```

## 🚀 Performance Optimization

### For Better Performance:
1. Use GPU version of FAISS: `pip install faiss-gpu`
2. Increase cache expiry in `app.py`: `CACHE_EXPIRY_HOURS = 48`
3. Use a larger embedding model for better accuracy
4. Deploy with Gunicorn: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`

## 🐛 Troubleshooting

### Issue: FAISS installation fails
**Solution**: 
- Windows: Use `faiss-cpu-windows`
- Mac M1/M2: Use `pip install faiss-cpu --no-cache`

### Issue: Model download is slow
**Solution**: Pre-download models:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### Issue: Port 5000 already in use
**Solution**: Change port in `app.py`:
```python
app.run(debug=True, port=8000)
```

## 📊 Technology Stack

- **Backend**: Flask, Python
- **ML/AI**: Sentence Transformers, FAISS
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Visualization**: Chart.js
- **Web Scraping**: BeautifulSoup4, Requests

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Indian Government for open data initiatives
- Sentence Transformers team for the embedding models
- FAISS team for the vector search library
- Flask community for the web framework

## 🎯 Future Enhancements

- [ ] Multi-language support (Hindi, Tamil, Telugu, etc.)
- [ ] Voice input/output capabilities
- [ ] Mobile application
- [ ] Integration with official government APIs
- [ ] User authentication and personalized recommendations
- [ ] Real-time notifications for new schemes
- [ ] Eligibility checker based on user profile
- [ ] Document upload for scheme application assistance

## 📧 Contact

For questions or support, please reach out to: your.email@example.com

---

**Made with ❤️ for making government schemes accessible to all Indians**
