# 🚀 Smart Profile Matcher

An AI-powered profile search and matching application that helps recruiters, hiring managers, and talent acquisition teams quickly identify the best candidate profiles based on skills, experience, job requirements, and keywords.

## 🌟 Features

✅ Intelligent profile matching using AI

✅ Semantic search for resumes and candidate profiles

✅ Skill-based ranking and scoring

✅ Modern and interactive UI

✅ Fast profile filtering and search

✅ Supports multiple profile formats

✅ Match percentage visualization

✅ Recruiter-friendly dashboard

## 📸 Application Overview

Smart Profile Matcher enables users to:

- Upload candidate profiles
- Search profiles using job descriptions
- Rank candidates based on relevance
- View detailed skill matching
- Analyze profile suitability instantly

## 🛠️ Technology Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Sentence Transformers
- FAISS (Optional)
- OpenAI / Groq / LLM Integration
- HTML & CSS Custom Styling

## 📂 Project Structure

```text
smart-profile-matcher/
│
├── app.py
├── requirements.txt
├── profiles/
├── assets/
├── utils/
├── models/
├── .streamlit/
│   └── secrets.toml
└── README.md
```
```text
rag_assistant/
│
├── __init__.py
├── config.py
├── documents.py
├── embeddings.py
├── hybrid_index.py
├── llm.py
├── pipeline.py
│
storage/
│
├── index/
│
tests/
├── conftest.py
├── test_app.py
├── test_rag.py

└── README.md
```
## Presentation Architecture

Presentation Layer
│
├── Streamlit UI
├── Recruiter Dashboard
├── Resume Upload
└── Candidate Search

↓

Application Layer
│
├── RAGPipeline
├── Query Expansion
├── Candidate Ranking
└── Response Generation

↓

Retrieval Layer
│
├── FAISS Vector Search
├── BM25 Keyword Search
└── RRF Fusion

↓

AI Layer
│
├── Embeddings Model
├── Groq LLM
└── Semantic Matching

↓

Data Layer
│
├── Resume PDFs
├── OCR Extracted Text
├── Resume Chunks
└── Search Index

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/smart-profile-matcher.git

cd smart-profile-matcher
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🔑 Environment Configuration

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
OPENAI_API_KEY=your_api_key
```

Or use Streamlit Secrets:

```toml
GROQ_API_KEY="your_api_key"
```

## ▶️ Run Application

```bash
streamlit run app.py
```

Application will launch at:

```text
http://localhost:8501
```

## ☁️ Deploy on Streamlit Cloud

1. Push code to GitHub
2. Create Streamlit Cloud account
3. Connect GitHub repository
4. Configure Secrets

```toml
GROQ_API_KEY="your_api_key"
```

5. Deploy

## 🎯 Use Cases

### Recruitment Teams

- Resume screening
- Candidate ranking
- Skill gap analysis

### Human Resources

- Internal talent search
- Employee skill mapping

### Staffing Agencies

- Client requirement matching
- Candidate shortlisting

## 📊 Matching Criteria

Profiles are evaluated based on:

- Technical Skills
- Experience
- Certifications
- Domain Knowledge
- Job Description Relevance
- Keyword Similarity
- Semantic Similarity Scores

## 🔒 Security

- API keys stored securely using Streamlit Secrets
- No sensitive credentials exposed in code
- Supports private repository deployment

## 🚀 Future Enhancements

- PDF Resume Parsing
- LinkedIn Profile Integration
- Interview Question Generation
- Candidate Summary Generation
- Multi-language Support
- Advanced AI Ranking Models

## 🤝 Contributing

Contributions, feature requests, and suggestions are welcome.

Fork the repository and submit a pull request.

## 📜 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Lavanya S**

Data & AI Professional | Analytics | Machine Learning Enthusiast

---

⭐ If you find this project useful, please give it a star on GitHub.
