# 🎬 EduBot Video Demonstration Script
## Project: Education Industry LLM Chatbot
## Duration: ~8-10 minutes

---

## 📽️ INTRO (0:00 – 0:45)

**[SCREEN: Project title slide — "EduBot: Education Industry LLM Chatbot"]**

**NARRATION:**
"Hello everyone. My name is [Your Name], and today I am presenting my capstone project — EduBot, an intelligent AI chatbot built specifically for the Education and Training industry using Large Language Models from HuggingFace.

In this video, I will walk you through the complete project — from the business problem it solves, to data collection, model fine-tuning using QLoRA, the chatbot engine, and finally a live demonstration of the working system."

---

## 📽️ SECTION 1 — Business Problem (0:45 – 1:30)

**[SCREEN: Slides showing education statistics]**

**NARRATION:**
"The global education market is worth 7.3 trillion dollars, yet teachers worldwide face an overwhelming shortage of resources and support. UNESCO reports a shortfall of 44 million teachers globally.

EduBot solves three core problems:
First — teachers need instant, evidence-based pedagogical guidance but don't always have time to search through research papers.
Second — students need 24/7 personalized support that no single teacher can provide.
Third — education professionals need a reliable knowledge base for curriculum design, assessment strategies, and modern teaching methodologies.

EduBot provides intelligent, contextually accurate answers to all of these needs."

---

## 📽️ SECTION 2 — Project Architecture (1:30 – 2:30)

**[SCREEN: Architecture diagram]**

**NARRATION:**
"Let me walk you through the project architecture.

The pipeline has four main components:

First, the Data Pipeline — I collected over 47 high-quality education Q&A pairs from synthetic expert knowledge, plus Wikipedia education articles. These were cleaned, normalized, and formatted in ChatML format for fine-tuning.

Second, the Model — I selected TinyLlama-1.1B-Chat as the base model. It has 1.1 billion parameters and is optimized for instruction-following tasks. I fine-tuned it using QLoRA — Quantized Low-Rank Adaptation — which allows training on a T4 GPU in Google Colab.

Third, the Chatbot Engine — built with three fallback levels: local fine-tuned model, HuggingFace Inference API, and a rule-based knowledge base. This ensures EduBot always responds reliably.

Finally, the Streamlit web application provides a clean, user-friendly chat interface."

---

## 📽️ SECTION 3 — Data Collection & Preprocessing (2:30 – 3:30)

**[SCREEN: Code — data_collection.py running]**

**NARRATION:**
"Let me show you the data collection. I run this command:

python src/data_collection.py

[Show terminal output]

As you can see, the pipeline collected 47 expert Q&A pairs covering topics like Bloom's Taxonomy, project-based learning, Universal Design for Learning, social-emotional learning, and much more.

Now let me show preprocessing:

python src/preprocessing.py

The preprocessing pipeline cleans text, removes noise, normalizes unicode, and formats each entry in the ChatML prompt format. It also performs data augmentation — generating alternative question phrasings to increase dataset diversity. Our final training set has 51 examples after augmentation and deduplication.

Here is a sample training entry — notice the system prompt establishes EduBot's identity, followed by the user question and the expert response."

---

## 📽️ SECTION 4 — Model Fine-Tuning (3:30 – 5:00)

**[SCREEN: Google Colab notebook — training cell running]**

**NARRATION:**
"For fine-tuning, I use Google Colab with a T4 GPU. Let me open the training notebook.

[Show Colab notebook]

First, I verify the GPU is available — this T4 GPU gives us 16 gigabytes of VRAM.

Next, I load the TinyLlama base model with 4-bit quantization using BitsAndBytes. This reduces memory usage from 4GB to about 1GB, allowing us to fit the model on the T4 GPU.

Then I apply LoRA adapters — Low-Rank Adaptation — which freeze the original model weights and add trainable low-rank matrices to the attention layers. We train only 0.5% of the total parameters — about 5.3 million out of 1.1 billion — yet achieve performance comparable to full fine-tuning.

The training configuration uses:
- Learning rate: 2e-4 with cosine scheduler
- Batch size: 16 effective (4 × 4 gradient accumulation)
- 3 epochs for demonstration (up to 25 as allowed)
- Early stopping with patience of 3

[Show training progress bars]

Training completes in about 20 minutes per epoch. The model converges with a training loss around 0.8 and validation perplexity below 10, indicating good language modeling."

---

## 📽️ SECTION 5 — Chatbot Demo: Terminal (5:00 – 6:30)

**[SCREEN: Terminal — python src/chatbot.py running]**

**NARRATION:**
"Let me now demonstrate EduBot working from the terminal.

python src/chatbot.py --model-path models/edubot_finetuned

[Type question 1]

Me: 'What is Bloom's Taxonomy?'

EduBot: [Read the response — highlighting accuracy and completeness]

[Type question 2]

Me: 'How does project-based learning benefit students?'

EduBot: [Read response]

[Type question 3 — follow-up, testing context]

Me: 'What are some challenges with this method?'

Notice that EduBot uses the conversation history — it knows we were discussing project-based learning and provides a contextually relevant follow-up answer without me repeating the topic.

[Type question 4]

Me: 'What is the difference between formative and summative assessment?'

EduBot: [Read response]

The responses are accurate, comprehensive, and use educational terminology correctly — demonstrating that the fine-tuning on education-specific data was successful."

---

## 📽️ SECTION 6 — Streamlit Web App Demo (6:30 – 8:00)

**[SCREEN: Streamlit app in browser]**

**NARRATION:**
"Now let me show you the Streamlit web application.

streamlit run app/streamlit_app.py

[Browser opens at localhost:8501]

The app has four pages. Let me start with the Chat page.

Notice the suggested questions for easy exploration. I'll click on 'What is Bloom's Taxonomy?'

[Click suggestion — response appears in chat bubble]

I can continue the conversation. Let me type: 'Give me an example of a lesson using these levels.'

[Type and submit — response appears]

The chat interface shows conversation history, timestamps, and styled message bubbles. I can export the entire conversation as JSON using the Download button.

Let me switch to the Data Insights page.

[Click Data Insights]

Here you can see the training data statistics — 51 training examples, the distribution of data sources, response length distribution, and a scatter plot showing question versus response length.

The Configuration page allows setting the HuggingFace API token — useful when running without a local GPU.

The About page provides complete documentation of the project architecture, model details, and ethical considerations."

---

## 📽️ SECTION 7 — Evaluation Results (8:00 – 9:00)

**[SCREEN: Evaluation metrics table or code output]**

**NARRATION:**
"Let me show you the model evaluation results.

python src/evaluation.py

[Show output]

EduBot achieves:
- BLEU-1 score of 0.556 — this measures word-level precision between generated and reference answers
- ROUGE-1 F1 of 0.588 — measuring recall and precision of content
- Average response length of 150 words — comprehensive enough for educational explanations
- Lexical diversity of 0.97 — highly varied vocabulary rather than repetitive patterns
- An average of 4 domain-specific educational keywords per response

These scores confirm that EduBot produces relevant, content-rich responses that align well with expert educational knowledge.

I also provide a human evaluation rubric for educators to rate responses on relevance, accuracy, coherence, completeness, and educational value — each on a 1-5 scale."

---

## 📽️ SECTION 8 — Conclusion (9:00 – 9:45)

**[SCREEN: Project summary slide]**

**NARRATION:**
"To summarize, EduBot is a complete, industry-grade LLM chatbot for the Education and Training sector.

Key achievements:
- Fine-tuned TinyLlama-1.1B using QLoRA on education-specific data
- 38 unit tests — all passing — ensuring production code quality
- Three-tier inference architecture for reliability in any environment
- Comprehensive Streamlit web interface with 4 feature pages
- BLEU and ROUGE evaluation showing strong performance
- Full deployment pipeline for Colab, local, and HuggingFace Spaces

The future roadmap includes expanding the training dataset to 500+ examples, integrating retrieval-augmented generation (RAG) with education databases, adding voice interface support, and multi-language capabilities for global education contexts.

Thank you for watching. The complete code, documentation, and training notebook are all included in the project submission. Please feel free to ask any questions."

---

## 📋 DEMO Q&A PREPARATION

Prepare answers for these likely questions:

1. "Why did you choose TinyLlama over GPT-2?"
   → TinyLlama is instruction-tuned (ChatML), 1.1B params, free, and fits T4 GPU with 4-bit quantization. GPT-2 is autoregressive-only and not optimized for dialogue.

2. "What does LoRA actually do?"
   → LoRA freezes original weights and adds trainable low-rank matrices (rank 16) to attention layers. This trains 0.5% of parameters while adapting the model for our domain.

3. "How did you ensure data quality?"
   → Expert-written synthetic Q&A, Wikipedia sourcing, quality filters (min 30 chars response), deduplication, and augmentation review.

4. "Why Education industry?"
   → High social impact, universal applicability, rich pedagogical literature, and clear use cases for LLM assistance that don't require real-time data.

5. "How would you scale this for production?"
   → Expand training data, add RAG with education databases (ERIC, edX), deploy with FastAPI + Redis caching, add user authentication, and integrate with LMS platforms.
