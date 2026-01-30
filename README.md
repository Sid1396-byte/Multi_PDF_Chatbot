# Multi_PDF_Chatbot
I wanted to solve the challenge of chatting with private documents without sending data to the cloud. Using Ollama (Gemma 4B) and Streamlit, I built a RAG (Retrieval-Augmented Generation) application that turns static PDFs into an interactive knowledge base.

🔥 Key Features: ✅ 100% Local Execution: No OpenAI API keys required. Runs on my own hardware using Ollama. ✅ Advanced Multi-Query Search: Solved the "semantic search ambiguity" problem. If you ask about two different topics at once, the system intelligently breaks it down into sub-queries to find all relevant info. ✅ Hallucination Control: Engineered "Strict Mode" prompts that force the AI to cite facts from the doc or admit when it doesn't know. ✅ Dynamic Context Windowing: Optimized chunking strategies (800 tokens) to balance context depth with model performance.

🧠 Technical Learnings: Building this taught me that Vector Search isn't magic. Standard embeddings often fail when a query covers multiple distinct concepts. By implementing a "Query Generation" step before retrieval, I significantly improved the accuracy of the answers.

Note: The code architecture and optimization were refined with the assistance of ChatGPT and Gemini to ensure best practices and efficient error handling.
