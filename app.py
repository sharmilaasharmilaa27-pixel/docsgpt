import streamlit as st

from config import (
    groq_client,
    GROQ_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
    MIN_SIMILARITY,
)

from embedder import embed_query
from store import query_index


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocsGPT",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(99, 102, 241, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(168, 85, 247, 0.10),
                transparent 30%
            ),
            #080b14;
        color: #f8fafc;
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #0b0f1a;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ---------- Hide Streamlit branding ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* ---------- Hero ---------- */

    .hero {
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(129,140,248,0.25);
        color: #a5b4fc;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.4px;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 52px;
        font-weight: 800;
        letter-spacing: -2px;
        margin: 0;
        background: linear-gradient(
            90deg,
            #ffffff,
            #c7d2fe,
            #e9d5ff
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        max-width: 680px;
        margin: 15px auto 0 auto;
        color: #94a3b8;
        font-size: 17px;
        line-height: 1.7;
    }

    /* ---------- Status ---------- */

    .status {
        display: flex;
        justify-content: center;
        margin: 20px 0 30px 0;
    }

    .status-pill {
        padding: 8px 16px;
        border-radius: 999px;
        background: rgba(34,197,94,0.08);
        border: 1px solid rgba(34,197,94,0.2);
        color: #86efac;
        font-size: 13px;
    }

    /* ---------- Cards ---------- */

    .info-card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 12px;
    }

    .source-card {
        padding: 14px 16px;
        border-radius: 14px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
        margin: 8px 0;
    }

    .source-title {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 14px;
    }

    .source-meta {
        color: #64748b;
        font-size: 12px;
        margin-top: 5px;
    }

    /* ---------- Chat ---------- */

    [data-testid="stChatMessage"] {
        background: transparent;
        border: none;
    }

    [data-testid="stChatMessageContent"] {
        border-radius: 18px;
        padding: 16px 20px;
    }

    /* ---------- Input ---------- */

    [data-testid="stChatInput"] {
        border-radius: 18px;
    }

    /* ---------- Metrics ---------- */

    .metric-card {
        text-align: center;
        padding: 18px;
        border-radius: 16px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
    }

    .metric-number {
        font-size: 25px;
        font-weight: 700;
        color: #e0e7ff;
    }

    .metric-label {
        color: #64748b;
        font-size: 12px;
        margin-top: 4px;
    }

    /* ---------- Welcome suggestions ---------- */

    .suggestion {
        padding: 15px 18px;
        border-radius: 14px;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.06);
        color: #cbd5e1;
        margin-bottom: 10px;
        transition: 0.2s;
    }

    .suggestion:hover {
        border-color: rgba(129,140,248,0.35);
        background: rgba(99,102,241,0.06);
    }

    /* ---------- Divider ---------- */

    .soft-divider {
        height: 1px;
        background: rgba(255,255,255,0.07);
        margin: 30px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="font-size:28px;font-weight:800;">
            ✦ DocsGPT
        </div>
        <div style="color:#64748b;margin-top:5px;">
            Your document intelligence layer
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### 🧠 Knowledge Base")

    st.markdown(
        """
        <div class="info-card">

        <b>6 Documents</b><br>
        <span style="color:#64748b;">
        Curated Markdown knowledge
        </span>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Retrieval")

    show_chunks = st.checkbox(
        "🔎 Show retrieved chunks",
        value=False,
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=8,
        value=TOP_K,
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="
            color:#64748b;
            font-size:12px;
            line-height:1.6;
        ">
        <b style="color:#94a3b8;">Powered by</b><br>
        Gemini Embeddings<br>
        Pinecone Vector Search<br>
        Groq LLM<br>
        Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HERO
# ============================================================

st.html("""
<div style="
    text-align: center;
    padding: 45px 20px 20px 20px;
">

    <div style="
        display: inline-block;
        padding: 8px 16px;
        border-radius: 30px;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(129,140,248,0.30);
        color: #a5b4fc;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
    ">
        ✦ RETRIEVAL-AUGMENTED INTELLIGENCE
    </div>

    <h1 style="
        font-size: 52px;
        font-weight: 800;
        letter-spacing: -2px;
        margin: 20px 0 10px 0;
        color: #f8fafc;
    ">
        Ask your documents.
    </h1>

    <p style="
        max-width: 680px;
        margin: 0 auto;
        color: #94a3b8;
        font-size: 17px;
        line-height: 1.7;
    ">
        Search your personal knowledge base using semantic retrieval
        and get grounded answers with transparent citations.
    </p>

</div>
""")
# ============================================================
# WELCOME SCREEN
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#64748b;
            margin-bottom:20px;
        ">
            Try asking something like
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "☀️ What are the planets in the Solar System?",
            use_container_width=True
        ):
            st.session_state.selected_question = (
                "What are the planets in the Solar System?"
            )

    with col2:
        if st.button(
            "🧠 What is overfitting in machine learning?",
            use_container_width=True
        ):
            st.session_state.selected_question = (
                "What is overfitting in machine learning?"
            )

    with col3:
        if st.button(
            "💰 What is compound growth?",
            use_container_width=True
        ):
            st.session_state.selected_question = (
                "What is compound growth?"
            )

# ============================================================
# FUNCTIONS
# ============================================================

def retrieve_documents(question):

    query_vector = embed_query(question)

    results = query_index(
        query_vector,
        top_k=top_k,
    )

    return results.get("matches", [])


def build_context(matches):

    context = []

    for i, match in enumerate(matches, start=1):

        metadata = match.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "Unknown source",
        )

        text = metadata.get(
            "text",
            "",
        )

        context.append(
            f"[{i}] Source: {source}\n{text}"
        )

    return "\n\n".join(context)


def generate_answer(question, context):

    prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer using ONLY the provided document context.

Citation rules:

- Every factual claim needs a citation.
- Use [1], [2], [3], etc.
- The citation must correspond to the provided context.
- Never invent citations.

If the answer cannot be supported by the context, respond exactly:

I don't have that in the docs.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_tokens=700,
    )

    return response.choices[0].message.content


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER QUESTION
# ============================================================

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

question = st.chat_input(
    "Ask anything about your documents..."
)


if question:

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # Assistant
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching your knowledge base..."):

            try:

                matches = retrieve_documents(
                    question
                )

                strong_matches = [
                    match
                    for match in matches
                    if match.get(
                        "score",
                        0,
                    ) >= MIN_SIMILARITY
                ]

                # ==================================================
                # NO RELEVANT DOCUMENT
                # ==================================================

                if not strong_matches:

                    answer = (
                        "I don't have that in the docs."
                    )

                    st.markdown(
                        f"""
                        <div class="info-card">

                        ❌ <b>Outside my knowledge base</b>

                        <br><br>

                        I couldn't find enough relevant
                        information in the documents to answer
                        this question reliably.

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                # ==================================================
                # RELEVANT DOCUMENT FOUND
                # ==================================================

                else:

                    context = build_context(
                        strong_matches
                    )

                    answer = generate_answer(
                        question,
                        context,
                    )

                    st.markdown(answer)

                    # ------------------------------------------------
                    # Sources
                    # ------------------------------------------------

                    st.markdown(
                        '<div class="soft-divider"></div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        "### 📚 Sources"
                    )

                    sources = []

                    for match in strong_matches:

                        source = match.get(
                            "metadata",
                            {},
                        ).get(
                            "source",
                            "Unknown source",
                        )

                        if source not in sources:
                            sources.append(source)

                    for i, source in enumerate(
                        sources,
                        start=1,
                    ):

                        st.markdown(
                            f"""
                            <div class="source-card">

                            <span style="
                                color:#a5b4fc;
                                font-weight:700;
                            ">
                            [{i}]
                            </span>

                            <span class="source-title">
                            &nbsp; {source}
                            </span>

                            <div class="source-meta">
                            Retrieved from Pinecone
                            </div>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # ------------------------------------------------
                    # Retrieval Debugger
                    # ------------------------------------------------

                    if show_chunks:

                        st.markdown(
                            '<div class="soft-divider"></div>',
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            "### 🔬 Retrieval Debugger"
                        )

                        st.caption(
                            "These are the chunks retrieved "
                            "before the answer was generated."
                        )

                        for i, match in enumerate(
                            strong_matches,
                            start=1,
                        ):

                            metadata = match.get(
                                "metadata",
                                {},
                            )

                            source = metadata.get(
                                "source",
                                "Unknown",
                            )

                            text = metadata.get(
                                "text",
                                "",
                            )

                            score = match.get(
                                "score",
                                0,
                            )

                            with st.expander(
                                f"[{i}] {source}   •   similarity {score:.3f}"
                            ):

                                st.progress(
                                    min(
                                        max(
                                            float(score),
                                            0.0,
                                        ),
                                        1.0,
                                    )
                                )

                                st.markdown(text)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

            except Exception as error:

                st.error(
                    "Something went wrong while processing your question."
                )

                st.exception(error)