# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------

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
# STYLING
# ---------------------------------------------------

st.markdown("""
<style>

.hero {
    padding:35px;
    border-radius:25px;
    background:linear-gradient(135deg,#0f172a,#2563eb,#7c3aed);
    color:white;
    text-align:center;
    margin-bottom:20px;
}

.result-card{
    background:white;
    padding:20px;
    border-radius:15px;
    border-left:5px solid #2563eb;
    box-shadow:0px 2px 8px rgba(0,0,0,0.1);
}

.stButton button{
    width:100%;
    border-radius:10px;
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
# HELPERS
# ---------------------------------------------------

def index_status():

    if len(pipeline.index) == 0:
        return "No profiles indexed."

    return (
        f"✅ {len(pipeline.index)} profile chunks "
        f"from {len(pipeline.index.sources)} files"
    )


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.title("🎯 Recruiter Dashboard")

    st.success(index_status())

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload Candidate Profiles",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    top_k = st.slider(
        "Top Matches",
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
        st.success("Index cleared")

    st.divider()

    st.markdown("""
### 🚀 Quick Tips

• Find Data Engineers

• Find Azure Engineers

• Compare Power BI Candidates

• Search AWS Skills

• Rank AI Candidates
""")


# ---------------------------------------------------
# HERO
# ---------------------------------------------------

st.markdown("""
<div class="hero">

<h1>🎯 Smart Profile Matcher</h1>

<h4>AI-Powered Talent Intelligence Platform</h4>

<p>
Upload resumes • Match talent • Rank candidates • Generate recruiter insights
</p>

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# KPIs
# ---------------------------------------------------

total_profiles = (
    len(pipeline.index.sources)
    if hasattr(pipeline.index, "sources")
    else 0
)

total_chunks = len(pipeline.index)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("📄 Profiles", total_profiles)

with c2:
    st.metric("🧠 Chunks", total_chunks)

with c3:
    st.metric("🎯 Top Results", top_k)

with c4:
    st.metric("🤖 AI Status", "Online")


# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Candidate Search",
        "📊 Analytics",
        "⚙️ Platform"
    ]
)


# ---------------------------------------------------
# SEARCH TAB
# ---------------------------------------------------

with tab1:

    st.subheader("💡 Popular Recruiter Searches")

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        if st.button("👨‍💻 Data Engineer"):
            # st.session_state.quick_query = \
            #     "Find Data Engineers"
            button_prompt  = "Find Data Engineers"

    with q2:
        if st.button("📊 Power BI"):
            st.session_state.quick_query = \
                "Find Power BI Developers"

    with q3:
        if st.button("☁️ Azure"):
            st.session_state.quick_query = \
                "Find Azure Engineers"

    with q4:
        if st.button("🤖 AI"):
            st.session_state.quick_query = \
                "Find AI Engineers"

    st.divider()

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    default_prompt = st.session_state.get(
        "quick_query",
        ""
    )

    chat_prompt  = st.chat_input(
        "Paste a job description or ask a recruiter question..."
    )
    prompt = button_prompt or chat_prompt
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

                st.markdown(
                    f"""
                    <div class="result-card">
                    {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

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

            st.error(
                f"LLM Error: {ex}"
            )

        except Exception as ex:

            st.error(
                f"Error: {ex}"
            )


# ---------------------------------------------------
# ANALYTICS TAB
# ---------------------------------------------------

with tab2:

    st.subheader("📊 Talent Analytics")

    a1, a2, a3 = st.columns(3)

    with a1:
        st.success(
            f"Profiles Indexed: {total_profiles}"
        )

    with a2:
        st.info(
            f"Profile Chunks: {total_chunks}"
        )

    with a3:
        st.warning(
            f"Top K Results: {top_k}"
        )

    st.divider()

    st.info("""
Upcoming Features:

• Skill Distribution Dashboard

• Candidate Leaderboard

• Hiring Funnel Analytics

• Experience Analytics

• Department Insights
""")


# ---------------------------------------------------
# PLATFORM TAB
# ---------------------------------------------------

with tab3:

    st.subheader(
        "🚀 Smart Profile Matcher"
    )

    st.markdown("""
### Platform Features

✅ Resume Parsing

✅ OCR Resume Processing

✅ Semantic Search

✅ Hybrid RAG Retrieval

✅ Skill Matching

✅ Experience Analysis

✅ Candidate Ranking

✅ Recruiter Recommendations

✅ Explainable AI
""")


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.markdown("""
<div style="
padding:25px;
border-radius:15px;
background:linear-gradient(90deg,#0f172a,#2563eb);
color:white;
text-align:center">

<h3>🎯 Smart Profile Matcher</h3>

<p>
AI Resume Intelligence • Talent Analytics • Semantic Candidate Search
</p>

</div>
""", unsafe_allow_html=True)