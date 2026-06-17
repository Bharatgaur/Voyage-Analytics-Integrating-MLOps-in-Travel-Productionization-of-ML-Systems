# 🎓 Interview & Viva Questions — EduBot LLM Project
## Complete Preparation Guide for LLM + NLP + MLOps Questions

---

## PART A — LLM & Transformers Core Questions

**Q1: What is a Large Language Model (LLM)?**
> A Large Language Model is a deep learning model, typically based on the transformer architecture, trained on massive text corpora to predict the next token in a sequence. LLMs like GPT, LLaMA, and TinyLlama develop emergent capabilities — translation, summarization, question-answering, reasoning — without being explicitly programmed for them. Their power comes from scale: billions of parameters trained on billions of tokens allow them to capture complex linguistic and world-knowledge patterns.

**Q2: Explain the transformer architecture.**
> The transformer architecture (Vaswani et al., 2017) uses self-attention mechanisms to relate every token in a sequence to every other token simultaneously, unlike RNNs that process sequentially. Key components: Multi-head self-attention (learns different relationship types in parallel), Feed-forward networks (non-linear transformation per position), Layer normalization and residual connections (training stability), and Positional encoding (injects order information). Decoder-only models like GPT/LLaMA use masked self-attention so each token only attends to previous tokens.

**Q3: What is the difference between pre-training and fine-tuning?**
> Pre-training trains a model from scratch on a large general corpus to learn language structure and world knowledge through self-supervised objectives (next-token prediction, masked language modeling). Fine-tuning takes a pre-trained model and continues training on a smaller, domain-specific dataset to specialize its behavior. Pre-training requires massive computation (weeks on hundreds of GPUs); fine-tuning can be done in hours on a single GPU. In EduBot, TinyLlama was pre-trained by its creators; we fine-tuned it on our education dataset.

**Q4: What is instruction tuning?**
> Instruction tuning is a fine-tuning approach where models are trained on (instruction → response) pairs to follow natural language instructions reliably. Unlike standard fine-tuning on raw text, instruction tuning explicitly teaches the model to interpret commands ("Explain X," "Write a Y," "Answer Q") and produce appropriate outputs. Models like InstructGPT, TinyLlama-Chat, and Llama-2-Chat are instruction-tuned versions of their base models.

**Q5: What is the attention mechanism and why does it matter?**
> Attention allows a model to dynamically weight the importance of different input tokens when generating each output token. The equation is: Attention(Q,K,V) = softmax(QKᵀ/√d_k)V, where Q (query), K (key), V (value) are linear projections of the input. This lets the model focus on "What is Bloom's Taxonomy?" when generating "taxonomy" rather than treating all preceding tokens equally. Multi-head attention runs this in parallel with different learned projections, capturing diverse relationship types simultaneously.

**Q6: What are tokens in NLP?**
> Tokens are the basic units a language model processes — typically sub-word pieces. Modern tokenizers (BPE, SentencePiece) split text into common word fragments: "education" → ["educ", "ation"]; "unhappy" → ["un", "happy"]. This allows the model to handle rare and unseen words by combining known subword units. A rough rule of thumb: 1 token ≈ 0.75 words in English. TinyLlama's context window accepts up to 2048 tokens.

---

## PART B — HuggingFace & Fine-Tuning Questions

**Q7: What is HuggingFace and what does it provide?**
> HuggingFace is an AI company and open-source platform providing: the Transformers library (standardized interface to 200,000+ pre-trained models), the Datasets library (easy access to thousands of NLP datasets), PEFT (parameter-efficient fine-tuning library), TRL (training LLMs with SFTTrainer, PPO, DPO), the Model Hub (sharing and downloading models), and Spaces (hosting ML demos). It has become the central infrastructure of the NLP research community.

**Q8: What is LoRA and why is it used?**
> LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes pre-trained model weights and injects trainable low-rank decomposition matrices into specific layers (typically attention: Q, K, V, O projections). Instead of updating all W parameters, it learns ΔW = A × B where A ∈ ℝ^(d×r) and B ∈ ℝ^(r×d), with r << d (rank is much smaller than dimension). For TinyLlama with r=16, we train ~5.3M parameters instead of 1.1B — 0.48% of the model — dramatically reducing compute, memory, and training time while achieving comparable performance.

**Q9: What is QLoRA?**
> QLoRA (Quantized LoRA) combines 4-bit quantization with LoRA. The base model weights are quantized to 4-bit precision (NF4 format via BitsAndBytes), reducing memory by ~75% compared to float16. LoRA adapters are kept in float16 for precision during training. This allows fine-tuning of 7B+ parameter models on a single consumer GPU. For EduBot, we used QLoRA on TinyLlama-1.1B: base model loads at ~600MB (4-bit) instead of 2.2GB (float16).

**Q10: What is SFTTrainer?**
> SFTTrainer (Supervised Fine-Tuning Trainer) from HuggingFace TRL simplifies instruction fine-tuning. It handles: automatic tokenization of text fields, response template masking (computing loss only on response tokens, not the prompt), sequence packing for efficiency, and integration with PEFT/LoRA. Compared to raw TrainingArguments, SFTTrainer is specifically optimized for the (instruction → response) supervised learning paradigm.

**Q11: Explain the difference between PEFT methods: LoRA vs Prompt Tuning vs Prefix Tuning.**
> LoRA adds trainable weight matrices inside the model layers — most widely used, best performance. Prompt Tuning prepends learned "soft" tokens to the input without changing model weights — very efficient but lower quality. Prefix Tuning prepends learned vectors to every attention layer's key-value pairs — intermediate between LoRA and prompt tuning. For domain adaptation like EduBot, LoRA consistently outperforms other PEFT methods on quality-efficiency tradeoff.

**Q12: What is 4-bit quantization?**
> Quantization reduces the precision of model weights from float32 (32 bits) or float16 (16 bits) to lower precision (8-bit, 4-bit, 2-bit), dramatically reducing memory. 4-bit quantization stores each weight in 4 bits instead of 16 — a 4× memory reduction. NF4 (Normal Float 4) is the format used in QLoRA: it uses a normal distribution-optimal codebook for the 16 possible 4-bit values, minimizing information loss for weights that typically follow Gaussian distributions.

---

## PART C — NLP & Evaluation Questions

**Q13: What is BLEU score?**
> BLEU (Bilingual Evaluation Understudy) measures the n-gram precision of a generated text against one or more references. It computes: how many 1-grams, 2-grams, 3-grams, 4-grams from the hypothesis appear in the reference (clipped to avoid repeat rewarding), multiplied by a brevity penalty for short outputs. BLEU ranges 0-1 (or 0-100%). A BLEU-4 of 0.3+ is generally considered good for dialogue. Limitation: BLEU penalizes valid paraphrases and is known to correlate poorly with human judgment in open-ended generation.

**Q14: What is ROUGE score?**
> ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures overlap between generated and reference text, with a focus on recall. ROUGE-1 and ROUGE-2 measure 1-gram and 2-gram overlap F1. ROUGE-L measures the longest common subsequence F1. Unlike BLEU (precision-focused), ROUGE emphasizes recall — did the response cover the reference content? ROUGE is widely used for summarization evaluation and often correlates better with human judgment than BLEU.

**Q15: What is perplexity in language modeling?**
> Perplexity measures how well a language model predicts a test corpus — lower is better. Perplexity = exp(average cross-entropy loss). A perplexity of 10 means the model is, on average, as confused as if it had to choose uniformly among 10 equally likely options at each token. After fine-tuning EduBot, perplexity < 10 on validation data indicates the model has learned a confident, coherent distribution over the education domain.

**Q16: What is prompt engineering?**
> Prompt engineering is the practice of designing and optimizing text prompts to elicit the best possible responses from a language model without changing its weights. Techniques include: Zero-shot prompting (just asking), Few-shot prompting (providing examples in the prompt), Chain-of-Thought prompting (asking the model to reason step-by-step), System prompts (setting role/context), and Template engineering (standardized prompt formats like ChatML). For EduBot, we use a system prompt establishing educational expertise + 2 few-shot examples.

**Q17: What is the difference between zero-shot, one-shot, and few-shot prompting?**
> Zero-shot: No examples given — just the instruction ("What is Bloom's Taxonomy?"). One-shot: One example given before the actual question. Few-shot: Multiple examples (typically 3-10) demonstrating the desired format and style. Few-shot prompting significantly improves performance by showing the model exactly what kind of response is expected. EduBot uses 2 few-shot examples in the prompt for single-turn queries.

**Q18: What is tokenization and what are common tokenization algorithms?**
> Tokenization converts raw text to token IDs the model processes. Common algorithms: BPE (Byte Pair Encoding) — iteratively merges most frequent character pairs; used by GPT. WordPiece — similar to BPE but maximizes language model probability; used by BERT. SentencePiece — language-agnostic, handles Unicode; used by LLaMA/TinyLlama. Unigram — probabilistic model selecting optimal segmentation. All create a vocabulary (typically 32K-100K tokens) mapping subword pieces to integer IDs.

---

## PART D — Project-Specific Viva Questions

**Q19: Why did you choose the Education industry?**
> Education is universally impactful, does not have the ethical restrictions of healthcare or finance per current university guidelines, has a rich body of publicly available pedagogical knowledge, and presents clear, practical use cases for LLM assistance. The industry's pain points — teacher shortage, personalization gap, knowledge access — directly align with LLMs' strengths in information retrieval and natural language explanation.

**Q20: Why TinyLlama over GPT-2 or BERT?**
> GPT-2 is autoregressive but not instruction-tuned — it generates continuations, not structured answers. BERT is encoder-only and suited for classification, not generation. TinyLlama-Chat is decoder-only, instruction-tuned with ChatML format, 1.1B parameters (much larger than GPT-2's 124M-1.5B yet memory-efficient with 4-bit quantization), Apache 2.0 licensed, and specifically designed for conversational instruction following — exactly what EduBot needs.

**Q21: What is the ChatML format?**
> ChatML is a structured prompt format using special tokens to denote conversation roles: `<|system|>`, `<|user|>`, `<|assistant|>`. Each turn is tagged with its role, allowing the model to distinguish between the system context, user queries, and its own previous responses. TinyLlama-Chat was trained with ChatML, so using this format during fine-tuning and inference aligns with the model's pre-trained expectations, resulting in better response quality.

**Q22: How does your fallback system work?**
> EduBot uses three inference tiers: (1) If a fine-tuned local model exists, use it for best domain-specific quality. (2) If a HuggingFace API token is provided, use the Inference API — useful for Colab demos without loading the model locally. (3) Rule-based fallback — 25 education topics with expert pre-written responses matched by keyword. The fallback ensures EduBot always provides a useful educational response even if the LLM backend is unavailable.

**Q23: What are the limitations of your EduBot?**
> Key limitations: (1) Small training dataset (58 examples) — larger datasets would improve coverage. (2) Potential hallucination on topics outside the training distribution. (3) No real-time knowledge updates — static training data. (4) English-only — multilingual support is a future goal. (5) Limited multi-turn reasoning for very complex pedagogical discussions. (6) BLEU/ROUGE metrics are imperfect for open-ended generation — human evaluation is the gold standard.

**Q24: How would you improve EduBot in a production setting?**
> (1) Expand training data to 500+ examples including peer-reviewed papers from ERIC. (2) Implement Retrieval-Augmented Generation (RAG) linking to education databases for grounded, verifiable answers. (3) Add human feedback loop (RLHF/DPO) using teacher ratings to align responses with professional standards. (4) Deploy with FastAPI + Redis caching for high-throughput serving. (5) Monitor response quality with automated metrics and human review. (6) Add multilingual support for global educational institutions.

**Q25: Explain your data augmentation approach.**
> We applied template-based instruction paraphrasing: questions matching patterns like "What is X?" were rewritten using 8 alternative templates ("Explain X", "Can you describe X?", "Tell me about X.", etc.). This increases linguistic diversity without requiring new knowledge content. The augmented instructions map to the same expert responses, ensuring quality is maintained while expanding the model's ability to handle varied query formulations. This added 11 examples (23% increase) to the training set.

---

## PART E — Common Mistakes in LLM Projects

1. **Using too small a dataset** — Even with LoRA, you need 50+ high-quality examples minimum; more is always better.

2. **Ignoring data quality for quantity** — 50 expert Q&A pairs outperform 500 noisy, inconsistent examples.

3. **Forgetting the pad token** — GPT-style models don't have pad tokens by default; always set `tokenizer.pad_token = tokenizer.eos_token`.

4. **Not setting `use_cache = False`** — Causes conflicts with gradient checkpointing during fine-tuning.

5. **Over-fitting to training examples** — A training loss of 0.0 means memorization, not generalization; validate on held-out data.

6. **Prompt mismatch** — Fine-tuning with one template (Alpaca) but inferring with another (ChatML) degrades performance.

7. **Not evaluating qualitatively** — BLEU/ROUGE don't capture factual accuracy, pedagogical appropriateness, or response helpfulness.

8. **Skipping early stopping** — Without early stopping, models overfit; training for all 25 epochs without monitoring is risky.

9. **Hallucination acceptance** — Not including verification steps for factual claims in educational content can mislead users.

10. **Neglecting the deployment-inference gap** — A model that works in training mode may behave differently after merging LoRA adapters; always test after merge.

---

## PART F — Real-World Deployment Challenges

1. **GPU costs** — Running 1.1B+ parameter models requires expensive infrastructure; quantization and efficient serving (vLLM, llama.cpp) are essential.

2. **Latency** — Token-by-token generation is slow; streaming responses and caching frequent queries improves user experience.

3. **Context length limits** — Long conversations exceed the model's context window (2048 tokens for TinyLlama); implement sliding window or summarization.

4. **Factual hallucination** — LLMs confidently state incorrect information; RAG or human-in-the-loop review is necessary for high-stakes educational content.

5. **Prompt injection** — Users can attempt to override system prompts; input validation and output filtering are required.

6. **Scalability** — Serving thousands of concurrent users requires load balancing, model parallelism, and auto-scaling.

7. **Model drift** — Educational best practices evolve; models require periodic retraining with updated data.

8. **Evaluation in production** — Offline BLEU/ROUGE doesn't reflect real user satisfaction; A/B testing and user feedback mechanisms are essential.

9. **Legal and ethical compliance** — Educational AI must comply with FERPA, COPPA, GDPR; careful data handling and privacy-by-design are mandatory.

10. **Accessibility** — Ensuring the chatbot is usable by people with disabilities (screen reader compatibility, keyboard navigation) is both ethical and often legally required.
