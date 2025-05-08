import os
from typing import List
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import re

# Load environment variables
load_dotenv()

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    video_id = None
    if 'youtube.com' in url:
        video_id = re.search(r'v=([^&]+)', url).group(1)
    elif 'youtu.be' in url:
        video_id = url.split('/')[-1]
    return video_id

def get_transcript(video_id: str) -> str:
    """Fetch transcript from YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

def create_embeddings(text: str) -> Chroma:
    """Create embeddings and store in ChromaDB."""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Store in ChromaDB
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectorstore

def setup_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    """Setup QA chain with OpenAI."""
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    return qa_chain

def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY in the .env file")
        return

    # Get YouTube URL from user
    video_url = input("Enter YouTube video URL: ")
    video_id = extract_video_id(video_url)
    
    if not video_id:
        print("Invalid YouTube URL")
        return

    # Get transcript
    print("Fetching transcript...")
    transcript = get_transcript(video_id)
    if not transcript:
        return

    # Create embeddings
    print("Creating embeddings...")
    vectorstore = create_embeddings(transcript)

    # Setup QA chain
    qa_chain = setup_qa_chain(vectorstore)

    # Interactive QA loop
    print("\nYou can now ask questions about the video. Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
            
        try:
            answer = qa_chain.run(question)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"Error getting answer: {e}")

if __name__ == "__main__":
    main() 