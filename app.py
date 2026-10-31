# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------
import pandas as pd
# import plotly.express as px
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

            files = []

            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(file.name).suffix
                ) as tmp:
                    tmp.write(file.read())

                    files.append(
                        {
                            "path": Path(tmp.name),
                            "source_name": file.name
                        }
                    )

            result = pipeline.ingest(files)

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
    st.metric("👥 Profiles Indexed", total_profiles)

with c2:
    st.metric("⭐ Best Match Score", total_chunks)

with c3:
    st.metric("🏅 Candidate Matches", top_k)

with c4:
    st.metric("🤖 AI Recruiter", "Online")


# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Candidate Search",
        "🎯 Talent Intelligence",
        "⚙️ Platform"
    ]
)


# ---------------------------------------------------
# SEARCH TAB
# ---------------------------------------------------
with tab1:

    st.subheader("💡 Popular Recruiter Searches")

    button_prompt = None

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        if st.button("👨‍💻 Data Engineer"):
            button_prompt = (
                "Find the best matching Data Engineer candidates and rank them"
            )

    with q2:
        if st.button("📊 Power BI"):
            button_prompt = (
                "Find candidates with strong Power BI experience"
            )

    with q3:
        if st.button("☁️ Azure"):
            button_prompt = (
                "Find Azure Engineers and rank them"
            )

    with q4:
        if st.button("🤖 AI"):
            button_prompt = (
                "Find AI and Machine Learning Engineers"
            )

    st.divider()

    # Display chat history
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Manual user input
    chat_prompt = st.chat_input(
        "Paste a job description or ask a recruiter question..."
    )

    # Use button click or manual input
    prompt = button_prompt or chat_prompt

    if prompt:

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        try:

            with st.spinner("🔍 Matching profiles..."):

                result = pipeline.answer(
                    prompt,
                    st.session_state.messages,
                    top_k
                )
            st.session_state.last_result = result
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

                if result.hits:

                    st.success(
                        f"✅ Found {len(result.hits)} matching profile results"
                    )

                # sources = format_sources(result)
                #
                # if sources:
                #
                #     with st.expander(
                #         f"🎯 Matching Details ({len(result.hits)} Results)",
                #         expanded=False
                #     ):
                #         st.markdown(sources)

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
# Talent Intelligence TAB
# ---------------------------------------------------
COMMON_SKILLS = [
    "Python",
    "SQL",
    "Azure",
    "AWS",
    "Power BI",
    "Tableau",
    "Snowflake",
    "Databricks",
    "Spark",
    "Airflow",
    "Machine Learning",
    "TensorFlow",
    "Pandas",
    "ETL",
    "Data Engineering"
]
from collections import defaultdict

skill_profiles = defaultdict(set)

if hasattr(pipeline.index, "chunks"):

    for chunk in pipeline.index.chunks:

        text = chunk.text.lower()

        for skill in COMMON_SKILLS:

            if skill.lower() in text:
                skill_profiles[skill].add(chunk.source)

skill_df = pd.DataFrame(
    [
        (skill, len(files))
        for skill, files in skill_profiles.items()
    ],
    columns=["Skill", "Candidates"]
)

skill_df = skill_df.sort_values(
    "Candidates",
    ascending=False
)

with tab2:

    st.subheader("🎯 Talent Intelligence")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Profiles Indexed",
            total_profiles
        )

    with col2:
        best_score = 0

        if hasattr(st.session_state, "last_result"):
            if st.session_state.last_result.hits:
                best_score = round(
                    st.session_state.last_result.hits[0].score * 100
                )

        st.metric(
            "⭐ Best Match Score",
            f"{best_score}%"
        )

    with col3:
        matches = 0

        if hasattr(st.session_state, "last_result"):
            matches = len(st.session_state.last_result.hits)

        st.metric(
            "🏅 Candidate Matches",
            matches
        )
    with col4:
        st.metric(
            "🤖 AI Recruiter",
            "Active"
        )
    st.divider()

    chart1, chart2 = st.columns(2)

    with chart1:

        st.subheader("📊 Top Skills in Talent Pool")

        if not skill_df.empty:

            st.bar_chart(
            skill_df,
            x="Skill",
            y="Candidates"
        )

        else:

            st.info(
            "Upload resumes to generate skill analytics."
        )

    with chart2:

        st.subheader("🏆 Recruitment Funnel")

        funnel_data = {
            "Stage": [
                "Profiles",
                "Shortlisted",
                "Interview",
                "Selected"
            ],
            "Count": [
                total_profiles,
                max(total_profiles - 1, 0),
                max(total_profiles - 2, 0),
                max(total_profiles - 3, 0)
            ]
        }

        st.bar_chart(
            data=funnel_data,
            x="Stage",
            y="Count"
        )

    st.divider()

    st.markdown("### 💡 Talent Insights")
    top_skill = "N/A"

    if not skill_df.empty:
        top_skill = skill_df.iloc[0]["Skill"]

    st.success(
    f"""
✅ {total_profiles} candidate profiles indexed

✅ Most common skill: {top_skill}

✅ AI-powered candidate matching available

✅ Semantic skill search enabled

✅ Recruiter-ready talent intelligence dashboard

✅ Candidate ranking and recommendation engine active
"""
    )

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