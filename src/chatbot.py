"""
src/chatbot.py
==============
EduBot Chatbot Engine — EduBot LLM Project
Industry: Education and Training

This module provides:
  1. EduBot class — the core chatbot engine
  2. Prompt engineering with system prompts + few-shot examples
  3. Conversation context management
  4. Response generation with quality controls
  5. Fallback to rule-based responses for reliability
  6. Hugging Face Inference API support (for Colab / no-GPU environments)

Usage
-----
    # With local fine-tuned model
    bot = EduBot(model_path="models/edubot_finetuned")
    response = bot.chat("What is Bloom's Taxonomy?")

    # With Hugging Face API
    bot = EduBot(use_api=True, api_token="hf_...")
    response = bot.chat("Explain project-based learning.")

Author: EduBot Project
"""

import os
import re
import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# PROMPT ENGINEERING
# ==============================================================================

SYSTEM_PROMPT = """You are EduBot, an expert AI assistant specializing in Education and Training. You have deep knowledge across:
- Pedagogy and teaching methodologies (Bloom's Taxonomy, Constructivism, UDL)
- Curriculum design and assessment strategies
- EdTech, e-learning platforms, and digital tools
- Special education and inclusive practices
- Higher education, K-12 education, and early childhood development
- Social-emotional learning (SEL) and student wellbeing
- Professional development for educators
- STEM education and 21st-century learning skills

You provide accurate, evidence-based, clear, and helpful answers. When you don't know something, you honestly say so and suggest resources. You always maintain a supportive, encouraging tone appropriate for an educational context."""


# ── Few-shot examples used in the prompt for quality guidance ────────────────
FEW_SHOT_EXAMPLES = [
    {
        "user": "What is Bloom's Taxonomy?",
        "assistant": "Bloom's Taxonomy is a hierarchical framework that classifies educational objectives into six cognitive levels: Remember, Understand, Apply, Analyze, Evaluate, and Create. It helps educators design lessons that progressively challenge students from basic recall to higher-order thinking skills."
    },
    {
        "user": "How can I help a struggling student?",
        "assistant": "There are several effective strategies to support struggling students: First, identify the specific challenge through diagnostic assessment. Then provide scaffolded support — break tasks into smaller steps, use graphic organizers, and offer worked examples. Build a supportive relationship so the student feels safe to ask questions. Connect with parents/guardians for a unified approach. Consider whether the student may need a referral for learning support services. Most importantly, maintain high expectations while adjusting your instructional approach."
    },
    {
        "user": "What are the benefits of collaborative learning?",
        "assistant": "Collaborative learning offers numerous benefits: students develop communication and teamwork skills essential for the workplace, they encounter diverse perspectives that deepen understanding, explaining concepts to peers reinforces their own learning (the protégé effect), it builds social connections that improve school belonging and motivation, and it prepares students for the collaborative nature of modern professional environments."
    }
]


def build_few_shot_prompt(query: str, conversation_history: List[Dict] = None) -> str:
    """
    Build a complete prompt with system context, few-shot examples,
    conversation history, and the current user query.

    Format: ChatML (<|system|>, <|user|>, <|assistant|>)

    Parameters
    ----------
    query                : current user question
    conversation_history : list of previous turns [{role, content}, ...]

    Returns
    -------
    Formatted prompt string
    """
    prompt_parts = []

    # ── System prompt ──────────────────────────────────────────────────────────
    prompt_parts.append(f"<|system|>\n{SYSTEM_PROMPT}")

    # ── Few-shot examples (prepend to first query only) ────────────────────────
    if not conversation_history:
        for example in FEW_SHOT_EXAMPLES[:2]:  # Use 2 examples to save tokens
            prompt_parts.append(f"<|user|>\n{example['user']}")
            prompt_parts.append(f"<|assistant|>\n{example['assistant']}")

    # ── Conversation history ────────────────────────────────────────────────────
    if conversation_history:
        # Include last N turns to manage context length
        recent_history = conversation_history[-6:]  # Last 3 exchanges (6 turns)
        for turn in recent_history:
            role    = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                prompt_parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}")

    # ── Current query ───────────────────────────────────────────────────────────
    prompt_parts.append(f"<|user|>\n{query}")
    prompt_parts.append("<|assistant|>")

    return "\n".join(prompt_parts)


def build_qa_prompt(query: str) -> str:
    """Simplified single-turn QA prompt (no conversation history)."""
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{query}\n"
        f"<|assistant|>\n"
    )


# ==============================================================================
# RULE-BASED FALLBACK RESPONSES
# ==============================================================================

# Maps keywords to pre-written expert responses for reliability
KNOWLEDGE_BASE: Dict[str, str] = {
    "bloom": "Bloom's Taxonomy classifies educational objectives into six hierarchical levels: Remember, Understand, Apply, Analyze, Evaluate, and Create. Teachers use it to design lessons that develop progressively higher-order thinking skills.",
    "constructivism": "Constructivism is a learning theory holding that learners actively build knowledge from experience rather than passively receiving information. Key theorists include Piaget (cognitive constructivism) and Vygotsky (social constructivism).",
    "differentiated instruction": "Differentiated instruction adjusts content, process, products, and environment to meet individual student needs based on readiness, interests, and learning profiles.",
    "project-based learning": "Project-Based Learning (PBL) is an instructional method where students learn by actively working on complex, real-world projects. It develops critical thinking, collaboration, communication, and creativity.",
    "formative assessment": "Formative assessment is ongoing evaluation during learning (quizzes, exit tickets, discussions) that provides feedback to guide instruction and student progress — unlike summative assessment which measures final achievement.",
    "scaffolding": "Scaffolding provides temporary, structured support as students learn new concepts, gradually removing support as competence grows. Based on Vygotsky's Zone of Proximal Development.",
    "flipped classroom": "In a flipped classroom, students study new content at home (videos, readings) and use class time for practice and problem-solving with teacher support, maximizing active learning during school.",
    "udl": "Universal Design for Learning (UDL) proactively designs flexible learning experiences with multiple means of representation, action/expression, and engagement to accommodate all learners.",
    "sel": "Social-Emotional Learning (SEL) develops five key competencies: self-awareness, self-management, social awareness, relationship skills, and responsible decision-making, improving both wellbeing and academic outcomes.",
    "growth mindset": "Growth mindset (Carol Dweck) is the belief that intelligence and abilities can develop through effort. Students with growth mindset embrace challenges, persist through setbacks, and view mistakes as learning opportunities.",
    "mooc": "MOOCs (Massive Open Online Courses) are online courses with open access via the internet, offered by platforms like Coursera, edX, and Khan Academy. They democratize education by making world-class learning globally accessible.",
    "stem": "STEM education integrates Science, Technology, Engineering, and Mathematics in an interdisciplinary, project-based approach that prepares students for high-demand technical careers and develops problem-solving skills.",
    "vygotsky": "Lev Vygotsky's social constructivism emphasizes that learning is fundamentally social. His key concept — the Zone of Proximal Development (ZPD) — describes the gap between what a learner can do alone vs. with guidance.",
    "piaget": "Jean Piaget's cognitive development theory describes four stages: Sensorimotor (0-2), Preoperational (2-7), Concrete Operational (7-11), and Formal Operational (12+). Children progress through these stages as they interact with the environment.",
    "inclusive education": "Inclusive education places students with disabilities alongside non-disabled peers in regular classrooms with appropriate support. Research shows benefits for all students — greater empathy for non-disabled students and improved outcomes for students with disabilities.",
    "gamification": "Gamification applies game elements (points, badges, leaderboards, levels) to educational contexts to increase motivation and engagement. Platforms like Kahoot!, Duolingo, and Prodigy use gamification to make learning more enjoyable.",
    "online learning": "Online learning (e-learning) delivers education through digital technologies. Advantages include flexibility, self-paced learning, global access to expertise, cost-effectiveness, and scalability. Challenges include digital equity, engagement, and academic integrity.",
    "early childhood": "Early childhood education (ages 0-8) is the most critical period for brain development. High-quality programs like Head Start build foundational cognitive, social, and emotional skills. Research shows a $7-12 return on every dollar invested.",
    "assessment": "Assessment measures student learning. Formative assessment provides ongoing feedback during learning; summative assessment measures achievement at the end of a unit. Effective education uses both types strategically.",
    "curriculum": "Curriculum design is the purposeful planning of what students will learn, how they will learn it, and how their learning will be assessed. Backward Design (Wiggins & McTighe) starts with desired outcomes and works backward to plan instruction.",
    "teaching": "Effective teaching involves deep subject knowledge, strong pedagogy, high expectations for all students, positive relationships, data-driven instruction, cultural responsiveness, and ongoing professional learning.",
    "learning style": "Learning styles theory (visual, auditory, kinesthetic) proposes that students learn best through their preferred modality. Note: While popular, current research does not strongly support learning styles for instructional design, but understanding student preferences can still inform engagement strategies.",
    "homework": "Research shows homework has a moderate positive effect in secondary school but minimal effect in primary school. Effective homework is purposeful, appropriately challenging, and moderate in amount (10 minutes per grade level).",
}


def get_rule_based_response(query: str) -> Optional[str]:
    """
    Check if the query matches a known topic in the knowledge base.
    Returns a pre-written response if a match is found, else None.
    """
    query_lower = query.lower()
    for keyword, response in KNOWLEDGE_BASE.items():
        if keyword.lower() in query_lower:
            return response
    return None


# ==============================================================================
# RESPONSE POST-PROCESSING
# ==============================================================================

def clean_response(text: str, prompt: str = "") -> str:
    """
    Clean the raw model output:
    - Remove the prompt portion if echoed
    - Strip special tokens
    - Remove repetitive text
    - Ensure clean ending
    """
    # Remove prompt if it appears in the output
    if prompt and text.startswith(prompt):
        text = text[len(prompt):]

    # Remove common special tokens
    for token in ["<|assistant|>", "<|user|>", "<|system|>", "<|endoftext|>", "</s>", "<s>"]:
        text = text.replace(token, "")

    # Take only the first response (stop at the next turn)
    # Split at next user turn indicator
    for stop in ["<|user|>", "\n### Instruction:", "\nHuman:", "\nUser:"]:
        if stop in text:
            text = text[:text.index(stop)]

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Ensure response ends cleanly (not mid-sentence)
    if text and not text[-1] in ".!?":
        # Find the last complete sentence
        for end in [". ", "! ", "? "]:
            last_idx = text.rfind(end)
            if last_idx > len(text) // 2:
                text = text[:last_idx + 1]
                break

    return text if text else "I'm sorry, I couldn't generate a relevant response. Please try rephrasing your question."


def validate_response(response: str, min_length: int = 20) -> bool:
    """Check if a response meets minimum quality standards."""
    if len(response.strip()) < min_length:
        return False
    if response.count("?") > 5:   # Likely asking questions rather than answering
        return False
    return True


# ==============================================================================
# MAIN CHATBOT CLASS
# ==============================================================================

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    role    : str    # "user" or "assistant"
    content : str
    timestamp: float = field(default_factory=time.time)


class EduBot:
    """
    EduBot — Education Industry Chatbot

    Supports three inference modes:
    1. Local fine-tuned model (production mode)
    2. Hugging Face Inference API (cloud mode, no GPU needed)
    3. Rule-based fallback (offline mode, always available)

    Parameters
    ----------
    model_path  : path to fine-tuned model directory
    use_api     : use HuggingFace Inference API instead of local model
    api_token   : HuggingFace API token (for use_api=True)
    api_model   : HuggingFace model ID for API calls
    temperature : sampling temperature (0.1=focused, 1.0=creative)
    max_tokens  : maximum new tokens to generate
    """

    def __init__(
        self,
        model_path  : str  = None,
        use_api     : bool = False,
        api_token   : str  = None,
        api_model   : str  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        temperature : float = 0.7,
        max_tokens  : int   = 512,
    ):
        self.model_path  = model_path
        self.use_api     = use_api
        self.api_token   = api_token or os.environ.get("HF_TOKEN", "")
        self.api_model   = api_model
        self.temperature = temperature
        self.max_tokens  = max_tokens

        self.model     = None
        self.tokenizer = None
        self.history   : List[ConversationTurn] = []

        self._mode = "fallback"  # Will be updated during initialization

        self._initialize()

    def _initialize(self) -> None:
        """Initialize the appropriate inference backend."""
        if self.use_api and self.api_token:
            self._mode = "api"
            logger.info("EduBot initialized in API mode (model: %s)", self.api_model)
        elif self.model_path and Path(self.model_path).exists():
            try:
                self._load_local_model()
                self._mode = "local"
                logger.info("EduBot initialized with local model: %s", self.model_path)
            except Exception as e:
                logger.warning("Local model load failed (%s). Using fallback mode.", e)
                self._mode = "fallback"
        else:
            self._mode = "fallback"
            logger.info("EduBot initialized in fallback (rule-based) mode")

    def _load_local_model(self) -> None:
        """Load local fine-tuned model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info("Loading local model from %s …", self.model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()
        logger.info("Local model loaded successfully")

    # ── INFERENCE METHODS ──────────────────────────────────────────────────────

    def _generate_local(self, prompt: str) -> str:
        """Generate response using local model."""
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens    = self.max_tokens,
                do_sample         = True,
                temperature       = self.temperature,
                top_p             = 0.9,
                top_k             = 50,
                repetition_penalty= 1.15,
                pad_token_id      = self.tokenizer.eos_token_id,
                eos_token_id      = self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (exclude the prompt)
        new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return response

    def _generate_api(self, prompt: str) -> str:
        """Generate response via HuggingFace Inference API."""
        import requests as req

        API_URL = f"https://api-inference.huggingface.co/models/{self.api_model}"
        headers = {"Authorization": f"Bearer {self.api_token}"}

        payload = {
            "inputs"    : prompt,
            "parameters": {
                "max_new_tokens"    : self.max_tokens,
                "temperature"       : self.temperature,
                "top_p"             : 0.9,
                "repetition_penalty": 1.15,
                "return_full_text"  : False,
            },
        }

        try:
            response = req.post(API_URL, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and result:
                    return result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    return result.get("generated_text", "")
            elif response.status_code == 503:
                return "The AI model is loading. Please wait a moment and try again."
            else:
                logger.warning("API returned status %d", response.status_code)
                return ""
        except Exception as e:
            logger.error("API call failed: %s", e)
            return ""

    def _generate_fallback(self, query: str) -> str:
        """Rule-based fallback response."""
        response = get_rule_based_response(query)
        if response:
            return response

        # Generic educational fallback
        return (
            "That's a great educational question! While I'm operating in offline mode "
            "and can't access the full AI model right now, I recommend exploring: "
            "1) Educational research databases like ERIC (education.gov), "
            "2) The CASEL website for SEL resources, "
            "3) Edutopia (edutopia.org) for teaching strategies, "
            "4) Khan Academy for subject-specific content. "
            "Please rephrase your question or try again when the full model is available."
        )

    # ── PUBLIC INTERFACE ───────────────────────────────────────────────────────

    def chat(self, query: str, use_history: bool = True) -> str:
        """
        Generate a response to the user's query.

        Parameters
        ----------
        query       : user's question or message
        use_history : whether to include conversation history in the prompt

        Returns
        -------
        str — the bot's response
        """
        if not query or not query.strip():
            return "Please ask me an education-related question! I'm here to help."

        query = query.strip()

        # Build prompt with conversation history
        history_turns = (
            [{"role": t.role, "content": t.content} for t in self.history[-10:]]
            if use_history else []
        )

        prompt = build_few_shot_prompt(query, history_turns)

        # Generate response using appropriate backend
        raw_response = ""
        if self._mode == "local":
            raw_response = self._generate_local(prompt)
        elif self._mode == "api":
            raw_response = self._generate_api(prompt)
        else:
            # Fallback mode — directly use rule-based without going through clean_response
            response = self._generate_fallback(query)
            self.history.append(ConversationTurn(role="user",      content=query))
            self.history.append(ConversationTurn(role="assistant", content=response))
            return response

        # Clean and validate model output
        response = clean_response(raw_response, prompt)

        # Fall back to rule-based if model response is inadequate
        if not validate_response(response):
            response = self._generate_fallback(query)

        # Update conversation history
        self.history.append(ConversationTurn(role="user",      content=query))
        self.history.append(ConversationTurn(role="assistant", content=response))

        return response

    def reset_history(self) -> None:
        """Clear conversation history to start a fresh session."""
        self.history = []
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict]:
        """Return conversation history as a list of dicts."""
        return [{"role": t.role, "content": t.content} for t in self.history]

    def get_mode(self) -> str:
        """Return the current inference mode."""
        return self._mode

    def get_stats(self) -> Dict:
        """Return session statistics."""
        user_turns = [t for t in self.history if t.role == "user"]
        return {
            "mode"              : self._mode,
            "total_turns"       : len(self.history),
            "user_messages"     : len(user_turns),
            "model"             : self.model_path or self.api_model,
        }

    def __repr__(self) -> str:
        return f"EduBot(mode={self._mode}, turns={len(self.history)})"


# ==============================================================================
# STANDALONE TERMINAL CHATBOT
# ==============================================================================

def run_terminal_chatbot(bot: EduBot) -> None:
    """
    Run EduBot as an interactive terminal chatbot.
    Type 'quit', 'exit', or 'bye' to end the session.
    Type 'reset' to clear conversation history.
    Type 'history' to view conversation history.
    """
    print("\n" + "=" * 60)
    print("  🎓 Welcome to EduBot — Education AI Assistant")
    print("  Industry: Education and Training")
    print(f"  Mode: {bot.get_mode().upper()}")
    print("=" * 60)
    print("  Ask me anything about education, teaching, or learning!")
    print("  Commands: 'reset' | 'history' | 'quit'")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nThank you for using EduBot! Happy learning! 🎓")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye", "q"]:
            print("\nEduBot: Thank you for learning with me! Goodbye! 🎓")
            stats = bot.get_stats()
            print(f"\nSession stats: {stats['user_messages']} questions answered")
            break

        if user_input.lower() == "reset":
            bot.reset_history()
            print("EduBot: Conversation history cleared. Fresh start! 📚\n")
            continue

        if user_input.lower() == "history":
            history = bot.get_history()
            if not history:
                print("EduBot: No conversation history yet.\n")
            else:
                print("\n--- Conversation History ---")
                for turn in history:
                    role = "You" if turn["role"] == "user" else "EduBot"
                    print(f"{role}: {turn['content'][:100]}…\n")
            continue

        # Generate and display response
        print("EduBot: ", end="", flush=True)
        response = bot.chat(user_input)
        print(response)
        print()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EduBot — Education Industry Chatbot")
    parser.add_argument("--model-path", type=str, default=None, help="Path to fine-tuned model")
    parser.add_argument("--use-api",    action="store_true",    help="Use HuggingFace Inference API")
    parser.add_argument("--api-token",  type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--api-model",  type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--temperature",type=float, default=0.7)
    args = parser.parse_args()

    # Initialize bot
    bot = EduBot(
        model_path  = args.model_path,
        use_api     = args.use_api,
        api_token   = args.api_token,
        api_model   = args.api_model,
        temperature = args.temperature,
    )

    # Quick demo
    demo_questions = [
        "What is Bloom's Taxonomy?",
        "How does project-based learning work?",
        "What strategies help students with learning disabilities?",
    ]

    print("\n🎓 EduBot Demo Responses:")
    print("=" * 60)
    for q in demo_questions:
        print(f"\nQ: {q}")
        r = bot.chat(q)
        print(f"A: {r[:300]}…" if len(r) > 300 else f"A: {r}")

    # Start interactive session
    bot.reset_history()
    run_terminal_chatbot(bot)
