"""
app/streamlit_app.py
====================
EduBot Streamlit Web Application
Industry: Education and Training

A complete, production-ready chatbot UI featuring:
  - Real-time chat interface
  - Conversation history with export
  - Model configuration panel
  - EDA visualizations of training data
  - About / documentation section

Run:
    streamlit run app/streamlit_app.py

Author: EduBot Project
"""

import os
import sys
import json
import time
import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "🎓 EduBot — Education AI Assistant",
    page_icon  = "🎓",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main chat container */
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #e3f2fd; margin: 0.3rem 0 0 0; }

    /* Chat message bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #1a73e8, #1565c0);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        text-align: right;
    }
    .bot-bubble {
        background: #f1f3f4;
        color: #202124;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 85%;
        border-left: 4px solid #1a73e8;
    }
    .timestamp {
        font-size: 0.7rem;
        color: #9aa0a6;
        margin-top: 0.2rem;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Sidebar styling */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a73e8;
        border-bottom: 2px solid #1a73e8;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }

    /* Suggestions */
    .suggestion-chip {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        cursor: pointer;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

def init_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        "messages"       : [],        # List of {role, content, timestamp}
        "bot"            : None,      # EduBot instance
        "bot_mode"       : "fallback",
        "total_questions": 0,
        "session_start"  : datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "api_token"      : "",
        "temperature"    : 0.7,
        "max_tokens"     : 300,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_bot():
    """Get or initialize the EduBot instance based on current settings."""
    if st.session_state.bot is None:
        try:
            from chatbot import EduBot
            model_path = str(ROOT_DIR / "models" / "edubot_finetuned")

            # Check if fine-tuned model exists
            if Path(model_path).exists() and list(Path(model_path).glob("*.bin")):
                bot = EduBot(model_path=model_path,
                             temperature=st.session_state.temperature,
                             max_tokens=st.session_state.max_tokens)
            elif st.session_state.api_token:
                bot = EduBot(use_api=True,
                             api_token=st.session_state.api_token,
                             temperature=st.session_state.temperature,
                             max_tokens=st.session_state.max_tokens)
            else:
                bot = EduBot()  # Fallback mode

            st.session_state.bot      = bot
            st.session_state.bot_mode = bot.get_mode()
        except ImportError:
            st.session_state.bot = None

    return st.session_state.bot


# ==============================================================================
# SIDEBAR
# ==============================================================================

def render_sidebar() -> str:
    """Render the sidebar and return the current page."""
    with st.sidebar:
        st.markdown("## 🎓 EduBot")
        st.markdown("*Education AI Assistant*")
        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigate",
            ["💬 Chat", "📊 Data Insights", "⚙️ Configuration", "📖 About"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Bot status
        bot = get_bot()
        if bot:
            mode = st.session_state.bot_mode
            mode_colors = {"local": "🟢", "api": "🟡", "fallback": "🔵"}
            mode_icon   = mode_colors.get(mode, "⚪")
            st.markdown(f"**Status:** {mode_icon} `{mode.upper()}`")
        else:
            st.markdown("**Status:** ⚪ `LOADING`")

        # Session stats
        st.markdown(f"**Questions:** {st.session_state.total_questions}")
        st.markdown(f"**Session:** {st.session_state.session_start}")

        st.markdown("---")

        # Quick settings
        st.markdown("**Quick Settings**")
        st.session_state.temperature = st.slider("Temperature", 0.1, 1.0, st.session_state.temperature, 0.1,
                                                   help="Higher = more creative responses")
        st.session_state.max_tokens  = st.slider("Max Tokens", 100, 600, st.session_state.max_tokens, 50)

        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.bot:
                st.session_state.bot.reset_history()
            st.rerun()

        # Export chat
        if st.session_state.messages:
            chat_export = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Export Chat",
                data=chat_export,
                file_name=f"edubot_chat_{datetime.date.today()}.json",
                mime="application/json",
                use_container_width=True
            )

    return page


# ==============================================================================
# PAGE 1 — CHAT
# ==============================================================================

SUGGESTED_QUESTIONS = [
    "What is Bloom's Taxonomy?",
    "Explain project-based learning",
    "How does flipped classroom work?",
    "What is differentiated instruction?",
    "What is the Zone of Proximal Development?",
    "How can AI improve personalized learning?",
    "What is Universal Design for Learning?",
    "How does gamification help students?",
    "What are the benefits of early childhood education?",
    "Explain social-emotional learning (SEL)",
]


def render_chat_page() -> None:
    """Render the main chat interface."""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 EduBot</h1>
        <p>Your intelligent Education & Training AI Assistant — Ask anything about pedagogy, EdTech, curriculum, and more!</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Suggested Questions ────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("**💡 Try asking:**")
        cols = st.columns(3)
        for i, q in enumerate(SUGGESTED_QUESTIONS[:6]):
            with cols[i % 3]:
                if st.button(q, key=f"sug_{i}", use_container_width=True):
                    # Add as user message and get response
                    _process_message(q)
                    st.rerun()

    # ── Chat History ──────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(msg["content"])
                    st.caption(msg.get("timestamp", ""))
            else:
                with st.chat_message("assistant", avatar="🎓"):
                    st.write(msg["content"])
                    st.caption(msg.get("timestamp", ""))

    # ── Input Box ──────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask me anything about education… (e.g., 'What is Bloom's Taxonomy?')"):
        _process_message(prompt)
        st.rerun()


def _process_message(prompt: str) -> None:
    """Process a user message and generate bot response."""
    timestamp = datetime.datetime.now().strftime("%H:%M")

    # Add user message
    st.session_state.messages.append({
        "role"     : "user",
        "content"  : prompt,
        "timestamp": timestamp
    })
    st.session_state.total_questions += 1

    # Generate response
    bot = get_bot()
    if bot:
        with st.spinner("🎓 EduBot is thinking…"):
            response = bot.chat(prompt)
    else:
        # Emergency fallback
        from chatbot import get_rule_based_response
        response = get_rule_based_response(prompt) or (
            "I'm initializing. Please try again in a moment, "
            "or ask about: Bloom's Taxonomy, project-based learning, "
            "differentiated instruction, flipped classroom, or SEL."
        )

    # Add bot response
    st.session_state.messages.append({
        "role"     : "assistant",
        "content"  : response,
        "timestamp": datetime.datetime.now().strftime("%H:%M")
    })


# ==============================================================================
# PAGE 2 — DATA INSIGHTS
# ==============================================================================

@st.cache_data
def load_training_data():
    """Load processed training data for visualization."""
    train_path = ROOT_DIR / "data" / "processed" / "education_train.json"
    if train_path.exists():
        return pd.read_json(str(train_path), orient="records")
    return None


def render_data_insights_page() -> None:
    """Render the training data analysis page."""
    st.title("📊 Training Data Insights")
    st.markdown("Explore the dataset used to train EduBot.")

    df = load_training_data()
    if df is None:
        st.warning("Training data not found. Please run the preprocessing pipeline first.")
        st.code("python src/preprocessing.py", language="bash")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Examples", f"{len(df):,}")
    col2.metric("Avg Response (words)", f"{df['response'].str.split().str.len().mean():.0f}")
    col3.metric("Avg Question (chars)", f"{df['instruction'].str.len().mean():.0f}")
    col4.metric("Unique Sources", df['source'].nunique() if 'source' in df.columns else "N/A")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        # Response length distribution
        df["resp_words"] = df["response"].str.split().str.len()
        fig1 = px.histogram(df, x="resp_words", nbins=30,
                            title="Response Length Distribution (words)",
                            color_discrete_sequence=["#1a73e8"],
                            labels={"resp_words": "Word Count"})
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        # Source distribution
        if "source" in df.columns:
            src_counts = df["source"].value_counts().reset_index()
            src_counts.columns = ["source", "count"]
            fig2 = px.pie(src_counts, values="count", names="source",
                          title="Data Source Distribution",
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          hole=0.35)
            st.plotly_chart(fig2, use_container_width=True)

    # Question length vs Response length scatter
    df["inst_words"] = df["instruction"].str.split().str.len()
    fig3 = px.scatter(df.sample(min(100, len(df))), x="inst_words", y="resp_words",
                      title="Question Length vs Response Length",
                      labels={"inst_words": "Question Words", "resp_words": "Response Words"},
                      opacity=0.7, color_discrete_sequence=["#0d47a1"])
    st.plotly_chart(fig3, use_container_width=True)

    # Sample data table
    with st.expander("📄 View Sample Training Data"):
        display_cols = ["instruction", "response", "source"] if "source" in df.columns else ["instruction", "response"]
        st.dataframe(df[display_cols].head(20), use_container_width=True)


# ==============================================================================
# PAGE 3 — CONFIGURATION
# ==============================================================================

def render_configuration_page() -> None:
    """Render model configuration and setup instructions."""
    st.title("⚙️ Configuration & Setup")

    tab1, tab2, tab3 = st.tabs(["🔑 API Settings", "🤖 Model Info", "🚀 Quick Start"])

    with tab1:
        st.subheader("HuggingFace API Configuration")
        st.markdown("Use HuggingFace Inference API to run EduBot without a local GPU.")

        api_token = st.text_input(
            "HuggingFace API Token",
            value=st.session_state.api_token,
            type="password",
            placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
            help="Get your token from https://huggingface.co/settings/tokens"
        )

        api_model = st.selectbox(
            "Select API Model",
            ["TinyLlama/TinyLlama-1.1B-Chat-v1.0",
             "microsoft/DialoGPT-medium",
             "distilgpt2"],
            help="Choose the model to use via API"
        )

        if st.button("💾 Save & Reinitialize Bot", type="primary"):
            st.session_state.api_token  = api_token
            st.session_state.bot        = None  # Force reinit
            if api_token:
                from chatbot import EduBot
                st.session_state.bot = EduBot(use_api=True, api_token=api_token, api_model=api_model)
                st.session_state.bot_mode = "api"
                st.success(f"✅ EduBot reinitalized in API mode with {api_model}")
            else:
                st.warning("Please enter an API token.")

    with tab2:
        st.subheader("Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Base Model**
            - TinyLlama-1.1B-Chat-v1.0
            - 1.1 Billion parameters
            - Chat/instruction-tuned
            - Apache 2.0 license

            **Fine-Tuning Method**
            - PEFT with LoRA (r=16, α=32)
            - 4-bit quantization (QLoRA)
            - SFTTrainer (TRL library)
            - ~0.5% of parameters trained
            """)
        with col2:
            st.markdown("""
            **Training Data**
            - 40+ education Q&A pairs (synthetic expert)
            - Wikipedia education articles
            - Augmented instruction variants
            - ChatML format

            **Hardware**
            - Google Colab T4 GPU (16GB)
            - Training time: ~30 min (3 epochs)
            - Max 25 epochs (project limit)
            """)

        # Training metrics (if available)
        metrics_path = ROOT_DIR / "models" / "edubot_finetuned" / "training_metrics.json"
        if metrics_path.exists():
            with open(str(metrics_path)) as f:
                metrics = json.load(f)
            st.subheader("Training Metrics")
            m_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()])
            st.dataframe(m_df, use_container_width=True)

    with tab3:
        st.subheader("Quick Start Guide")
        st.markdown("""
        ### Step 1: Install Dependencies
        ```bash
        pip install -r requirements.txt
        ```

        ### Step 2: Collect & Process Data
        ```bash
        python src/data_collection.py
        python src/preprocessing.py
        ```

        ### Step 3: Fine-Tune (Google Colab T4 GPU)
        ```bash
        # In Google Colab:
        !pip install -q transformers datasets peft trl bitsandbytes accelerate
        !python src/fine_tuning.py
        ```

        ### Step 4: Run Streamlit App
        ```bash
        streamlit run app/streamlit_app.py
        ```

        ### Step 5: Test the API
        ```bash
        python src/chatbot.py --model-path models/edubot_finetuned
        ```
        """)


# ==============================================================================
# PAGE 4 — ABOUT
# ==============================================================================

def render_about_page() -> None:
    """Render the about/documentation page."""
    st.title("📖 About EduBot")

    st.markdown("""
    ## 🎓 EduBot — Education Industry LLM Chatbot

    **EduBot** is an AI-powered conversational assistant built specifically for the **Education and Training** industry.
    It leverages state-of-the-art Large Language Models fine-tuned on education-specific data to answer questions
    about pedagogy, curriculum design, teaching strategies, EdTech, student development, and more.

    ---

    ### 🏗️ Project Architecture

    ```
    User Query
        ↓
    Prompt Engineering (System Prompt + Few-Shot Examples)
        ↓
    EduBot Engine (chatbot.py)
        ↓
    ┌─────────────────────────────────┐
    │  Inference Backend Selection    │
    │  1. Local Fine-Tuned Model      │ ← Best quality
    │  2. HuggingFace Inference API   │ ← Cloud option
    │  3. Rule-Based Fallback         │ ← Always available
    └─────────────────────────────────┘
        ↓
    Response Post-Processing & Quality Check
        ↓
    Response → User
    ```

    ---

    ### 📊 Industry Coverage

    EduBot can answer questions across all major education domains:

    | Domain | Example Topics |
    |--------|---------------|
    | Pedagogy | Bloom's Taxonomy, Constructivism, Scaffolding |
    | Curriculum | Backward Design, Assessment, Learning Objectives |
    | EdTech | MOOCs, Gamification, AI in Education |
    | Special Education | Inclusive Education, IEPs, UDL |
    | SEL | Growth Mindset, Social-Emotional Learning |
    | Early Childhood | Brain Development, Montessori, Head Start |
    | Higher Education | Problem-Based Learning, Flipped Classroom |

    ---

    ### 🤖 Model Stack

    | Component | Technology |
    |-----------|-----------|
    | Base Model | TinyLlama-1.1B-Chat-v1.0 |
    | Fine-Tuning | QLoRA (PEFT, 4-bit) |
    | Training Framework | HuggingFace TRL + Transformers |
    | Evaluation | BLEU, ROUGE, Perplexity |
    | Interface | Streamlit |
    | Deployment | Google Colab / HuggingFace Spaces |

    ---

    ### 👥 Target Users

    - **Teachers & Educators** — Quick pedagogical guidance and strategy ideas
    - **Students** — Understanding educational concepts and study strategies
    - **Education Researchers** — Literature on teaching effectiveness
    - **School Administrators** — Policy and curriculum information
    - **EdTech Developers** — Educational framework references
    - **Parents** — Understanding how their children learn

    ---

    ### ⚖️ Ethical Considerations

    - All training data is publicly available or synthetically generated
    - No personal student data is collected or used
    - The bot clearly identifies itself as an AI assistant
    - Recommendations are for informational purposes only; professional educators should exercise judgment
    - The system includes fallbacks to prevent harmful or misleading responses
    """)


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    init_session_state()
    page = render_sidebar()

    if page == "💬 Chat":
        render_chat_page()
    elif page == "📊 Data Insights":
        render_data_insights_page()
    elif page == "⚙️ Configuration":
        render_configuration_page()
    elif page == "📖 About":
        render_about_page()


if __name__ == "__main__":
    main()
