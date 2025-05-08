import os
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

class ProductReviewAnalyzer:
    def __init__(self):
        """Initialize the ProductReviewAnalyzer with necessary components."""
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore = None

    def load_reviews_from_csv(self, csv_path: str):
        """
        Load reviews from a CSV file and add them to the vector store.
        
        Args:
            csv_path: Path to the CSV file containing reviews
        """
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Convert DataFrame rows to review dictionaries
        reviews = []
        for _, row in df.iterrows():
            review = {
                "text": row['review_text'],
                "metadata": {
                    "product": row['product'],
                    "rating": int(row['rating']),
                    "category": row['category']
                }
            }
            reviews.append(review)
        
        # Add reviews to vector store
        self.add_reviews(reviews)

    def add_reviews(self, reviews: List[Dict[str, str]]):
        """
        Add product reviews to the vector store.
        
        Args:
            reviews: List of dictionaries containing review data
                    Each dict should have 'text' and 'metadata' keys
        """
        # Extract texts and metadata
        texts = [review['text'] for review in reviews]
        metadatas = [review['metadata'] for review in reviews]
        
        # Split texts into chunks
        splits = self.text_splitter.create_documents(texts, metadatas=metadatas)
        
        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
        else:
            self.vectorstore.add_documents(splits)

    def analyze_reviews(self, query: str) -> str:
        """
        Analyze reviews based on the given query.
        
        Args:
            query: The analysis query/question
            
        Returns:
            str: Analysis result
        """
        if not self.vectorstore:
            return "No reviews have been added yet."

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a product review analyst. Analyze the provided reviews 
            and answer the user's question. Be specific and provide insights based on the 
            review content. If you can't find relevant information, say so.
            
            Context: {context}"""),
            ("human", "{input}")
        ])

        # Create the document chain
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )

        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            combine_docs_chain=document_chain
        )

        # Execute the chain
        result = retrieval_chain.invoke({"input": query})
        return result["answer"]

def main():
    # Example usage
    analyzer = ProductReviewAnalyzer()
    
    # Load reviews from CSV
    analyzer.load_reviews_from_csv("reviews.csv")
    
    # Example analyses
    queries = [
        "What are the main points of feedback for Smartphone X?",
        "What are the common complaints about Laptop Pro?",
        "What are the strengths of Wireless Earbuds?",
        "What are the key features mentioned in Smart Watch reviews?",
        "What is the overall sentiment about Gaming Console?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        analysis = analyzer.analyze_reviews(query)
        print(f"Analysis: {analysis}")

if __name__ == "__main__":
    main() 