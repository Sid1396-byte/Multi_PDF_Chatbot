import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
import tempfile
import os

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="PDF AI Generator", page_icon="📝", layout="wide")
st.title("📝 PDF Q&A chatbot (Ollama + RAG)")
st.write("Upload PDFs. I will generate text, explanations, or summaries based **strictly** on their content.")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

# ---------------- PDF PROCESSING ----------------
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        temp_dir = tempfile.TemporaryDirectory()
        pdf_paths = []

        for uploaded_file in uploaded_files:
            path = os.path.join(temp_dir.name, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_paths.append(path)

        # Load PDFs
        docs = []
        for path in pdf_paths:
            docs += PyPDFLoader(path).load()

        st.success(f"Loaded {len(uploaded_files)} PDFs!")

        # Chunk size 800: Large enough to capture full concepts
        split_docs = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=200
        ).split_documents(docs)

        # LLM & Embeddings
        llm = ChatOllama(model="gemma3:4b")
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        # Vector Store
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # ---------------- GENERATION AREA ----------------
    st.markdown("### Topic or Instruction")
    question = st.text_input(
        "Enter a topic, question, or writing instruction:", 
        placeholder="e.g., 'Explain the functions of a clutch and how it relates to torque'"
    )

    if st.button("Generate Text") and question:
        with st.spinner("Researching and Writing..."):
            
            # --- STEP 1: MULTI-QUERY SEARCH (Crucial for mixed topics) ---
            # We ask the AI to break your request into search terms
            query_gen_prompt = f"""
            Act as a researcher. Break the user's request into specific search queries to find facts in a document.
            User Request: {question}
            Output ONLY the search queries, one per line.
            """
            generated_queries_response = llm.invoke(query_gen_prompt)
            queries = generated_queries_response.content.strip().split('\n')
            queries = [q.strip() for q in queries if q.strip()]

            # --- STEP 2: GATHER CONTEXT ---
            unique_docs = {}
            for q in queries:
                results = retriever.invoke(q)
                for doc in results:
                    unique_docs[doc.page_content] = doc

            # Limit context to avoid confusing the model
            final_docs = list(unique_docs.values())[:15]
            context_text = "\n\n".join([doc.page_content for doc in final_docs])

            # --- STEP 3: GENERATIVE PROMPT ---
            # This prompt tells the AI to be a WRITER, not just a robot.
            final_prompt = f"""
            You are a professional technical writer. Your goal is to write a comprehensive response based on the Context provided.
            
            CONTEXT:
            {context_text}
            
            USER INSTRUCTION: 
            {question}
            
            WRITING RULES:
            1. Use the Context to generate the text. Do not ignore the facts.
            2. You are free to structure the answer as a fluent article, summary, or explanation.
            3. Connect ideas smoothly (don't just list facts).
            4. If the user asks about multiple topics, use clear Headings to separate them.
            5. If the context is missing information, explicitly state: "The provided documents do not cover [topic]."
            """

            response = llm.invoke(final_prompt)

            st.markdown("### Generated Response:")
            st.write(response.content)
            
            # Transparency: Show what context was used
            with st.expander("View Source Context Used"):
                st.write(context_text)