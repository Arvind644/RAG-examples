import os
import logging
import json
from typing import List, Optional
import fitz  # PyMuPDF
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import glob
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.default_config = {
            "chunk_size": 500,  # Reduced chunk size for better context
            "chunk_overlap": 50,  # Reduced overlap
            "temperature": 0,
            "supported_extensions": [".pdf", ".txt", ".docx"],
            "vectorstore_path": "vectorstore",
            "model_name": "gpt-3.5-turbo"
        }
        self.config = self.load_config()

    def load_config(self) -> dict:
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            return self.default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.default_config

    def save_config(self):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

def extract_text_from_file(file_path: str) -> str:
    """Extracts text from various file types."""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            doc = fitz.open(file_path)
            text = ""
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                if page_text.strip():
                    text += f"\n--- Page {page_num} ---\n{page_text}\n"
            return text
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""

def create_vectorstore_from_files(file_paths: List[str], config: Config, force_recreate: bool = False) -> FAISS:
    """Creates or loads a FAISS vectorstore from files."""
    vectorstore_path = config.config["vectorstore_path"]
    
    if not force_recreate and os.path.exists(vectorstore_path):
        try:
            logger.info("Loading existing vectorstore...")
            embeddings = OpenAIEmbeddings()
            return FAISS.load_local(vectorstore_path, embeddings)
        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")

    logger.info("Creating new vectorstore...")
    all_text = ""
    for path in tqdm(file_paths, desc="Processing files"):
        text = extract_text_from_file(path)
        if text:
            all_text += f"\n=== Content from {os.path.basename(path)} ===\n{text}\n"
            logger.info(f"Successfully extracted {len(text)} characters from {path}")
        else:
            logger.warning(f"No text extracted from {path}")

    if not all_text.strip():
        raise ValueError("No text could be extracted from any of the provided files")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.config["chunk_size"],
        chunk_overlap=config.config["chunk_overlap"],
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = splitter.split_text(all_text)
    logger.info(f"Created {len(chunks)} text chunks")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    
    try:
        os.makedirs(vectorstore_path, exist_ok=True)
        vectorstore.save_local(vectorstore_path)
        logger.info("Vectorstore saved successfully")
    except Exception as e:
        logger.error(f"Error saving vectorstore: {e}")

    return vectorstore

def get_supported_files(directory: str, config: Config) -> List[str]:
    """Get all supported files from a directory."""
    files = []
    for ext in config.config["supported_extensions"]:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    return files

def start_terminal_chat(qa_chain: RetrievalQA):
    """Starts an enhanced terminal-based chat loop."""
    print("\n🔍 Ask me anything about the documents (type 'exit' to quit):")
    print("Commands:")
    print("  /clear - Clear chat history")
    print("  /exit  - Exit the program")
    print("  /help  - Show this help message\n")

    chat_history = []
    
    while True:
        query = input("🧠 You: ").strip()
        
        if query.lower() in ["exit", "quit", "/exit"]:
            print("👋 Goodbye!")
            break
        elif query.lower() == "/clear":
            chat_history.clear()
            print("Chat history cleared!")
            continue
        elif query.lower() == "/help":
            print("\nAvailable commands:")
            print("  /clear - Clear chat history")
            print("  /exit  - Exit the program")
            print("  /help  - Show this help message\n")
            continue

        try:
            result = qa_chain.invoke({"query": query})
            chat_history.append((query, result["result"]))
            print(f"\n🤖 Answer: {result['result']}\n")
            
            # Print source information if available
            if "source_documents" in result:
                print("Sources:")
                for i, doc in enumerate(result["source_documents"], 1):
                    print(f"{i}. {doc.page_content[:200]}...")
                print()
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print("❌ Sorry, I encountered an error processing your query.")

def main():
    print("📄 Document Chatbot (Terminal Edition)\n")
    
    config = Config()
    
    while True:
        print("Options:")
        print("1. Process specific files")
        print("2. Process all files in a directory")
        print("3. Use existing vectorstore")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == "4":
            print("👋 Goodbye!")
            break
            
        try:
            if choice == "1":
                file_input = input("Enter file paths (comma-separated): ").strip()
                file_paths = [p.strip() for p in file_input.split(",") if os.path.exists(p.strip())]
            elif choice == "2":
                directory = input("Enter directory path: ").strip()
                if not os.path.exists(directory):
                    print("⚠️ Directory not found!")
                    continue
                file_paths = get_supported_files(directory, config)
            elif choice == "3":
                if not os.path.exists(config.config["vectorstore_path"]):
                    print("⚠️ No existing vectorstore found!")
                    continue
                file_paths = []
            else:
                print("⚠️ Invalid option!")
                continue

            if not file_paths and choice != "3":
                print("⚠️ No valid files found. Please check file paths.")
                continue

            print("⏳ Processing documents...")
            vectorstore = create_vectorstore_from_files(
                file_paths, 
                config,
                force_recreate=(choice != "3")
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    temperature=config.config["temperature"],
                    model_name=config.config["model_name"]
                ),
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            print("✅ Ready to chat!")
            start_terminal_chat(qa_chain)
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
