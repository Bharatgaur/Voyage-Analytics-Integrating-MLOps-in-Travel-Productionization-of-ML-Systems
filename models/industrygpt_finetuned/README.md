This directory holds the QLoRA fine-tuned model artifacts after running src/fine_tuning.py or notebooks/edubot_colab_training.ipynb.

Expected contents after training:
- adapter_config.json
- adapter_model.safetensors
- tokenizer.json
- tokenizer_config.json
- special_tokens_map.json

These files are excluded from version control via .gitignore due to size. Train the model locally or on Google Colab to populate this folder before running the chatbot with --model-path models/industrygpt_finetuned.
