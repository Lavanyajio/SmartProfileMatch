"""
Smart Profile Matcher
Run:
streamlit run streamlit_app.py
"""

import tempfile
from pathlib import Path
import streamlit as st

from rag_assistant.config import settings
from rag_assistant.pipeline import RAGPipeline, format_sources
from rag_assistant.llm import LLMError

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Smart Profile Matcher",
    page_icon="🎯",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background: #f8fafc;
}

.hero {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    color: white;
    margin-bottom: 20px;
}

.metric-card {
    background:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0 2px 10px rgba(0,0,0,.1);
}

.stButton button {
    width:100%;
    border-radius:10px;
}

.chat-box {
    background:#ffffff;
    padding:15px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

pipeline = st.session_state.pipeline

# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------

def index_status():

    if len(pipeline.index) == 0:
        return "No profiles indexed."

    return (
        f"✅ {len(pipeline.index)} profile chunks "
        f"from {len(pipeline.index.sources)} files"
    )


# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------

st.markdown("""
<div class="hero">
<h1>🎯 Smart Profile Matcher</h1>

<h3>AI-Powered Talent Intelligence Platform</h3>

<p>
Match resumes against job descriptions using semantic search,
vector embeddings, AI ranking and explainable recommendations.
</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# KPI CARDS
# ---------------------------------------------------

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "⚡ AI Matching",
        "Semantic Ranking"
    )

with c2:
    st.metric(
        "📊 Skill Analysis",
        "Gap Detection"
    )

with c3:
    st.metric(
        "🎯 Resume Screening",
        "Instant Insights"
    )

st.divider()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.header("📄 Resume Management")

    uploaded_files = st.file_uploader(
        "Upload Candidate Resumes",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    top_k = st.slider(
        "🎯 Top Matching Profiles",
        2,
        10,
        settings.top_k
    )

    if st.button("📥 Index Profiles"):

        if uploaded_files:

            paths = []

            for file in uploaded_files:

                suffix = Path(file.name).suffix

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as tmp:

                    tmp.write(file.read())
                    paths.append(Path(tmp.name))

            result = pipeline.ingest(paths)

            st.success(result)

    if st.button("🗑️ Clear Index"):

        pipeline.reset()
        st.success("Index cleared successfully.")

    st.markdown("---")

    st.subheader("📊 Status")
    st.info(index_status())

# ---------------------------------------------------
# CAPABILITIES
# ---------------------------------------------------

with st.expander(
        "📊 Matching Capabilities",
        expanded=False):

    st.markdown("""
✅ Resume Parsing

✅ Skill Extraction

✅ Job Description Analysis

✅ Semantic Candidate Ranking

✅ Experience Matching

✅ Skill Gap Identification

✅ Explainable AI Recommendations

✅ Candidate Comparison
""")

# ---------------------------------------------------
# SAMPLE QUESTIONS
# ---------------------------------------------------

st.subheader("💡 Recruiter Queries")

samples = [
    "Find candidates suitable for Senior Data Engineer.",
    "Rank candidates for Machine Learning Engineer.",
    "Compare Azure skills across profiles.",
    "Who is strongest in Power BI?",
    "Identify missing skills for this role."
]

for s in samples:
    st.write("•", s)

st.divider()

# ---------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------

prompt = st.chat_input(
    "Paste a job description or ask about candidate skills..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    try:

        result = pipeline.answer(
            prompt,
            st.session_state.messages,
            top_k
        )

        answer = result.text

        with st.chat_message("assistant"):
            st.markdown(answer)

            sources = format_sources(result)

            if sources:
                with st.expander(
                    f"🎯 Matching Details ({len(result.hits)} Results)"
                ):
                    st.markdown(sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    except LLMError as ex:

        st.error(f"LLM Error: {ex}")

    except Exception as ex:

        st.error(f"Error: {ex}")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.markdown("""
<center>

### Smart Profile Matcher

AI-Powered Resume Intelligence Platform

Built with FAISS • BM25 • Hugging Face • Groq • Streamlit

</center>
""", unsafe_allow_html=True)