# Product Review Analyzer

A Python-based application that uses LangChain and Chroma DB to analyze product reviews using RAG (Retrieval Augmented Generation).

## Features

- Store and manage product reviews in a vector database
- Analyze reviews using natural language queries
- Extract insights and patterns from review data
- Support for metadata (product name, rating, category, etc.)

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Import and initialize the analyzer:
   ```python
   from review_analyzer import ProductReviewAnalyzer
   
   analyzer = ProductReviewAnalyzer()
   ```

2. Add reviews:
   ```python
   reviews = [
       {
           "text": "Your review text here",
           "metadata": {
               "product": "Product Name",
               "rating": 5,
               "category": "Category"
           }
       }
   ]
   analyzer.add_reviews(reviews)
   ```

3. Analyze reviews:
   ```python
   query = "What are the main points of feedback for Product Name?"
   analysis = analyzer.analyze_reviews(query)
   print(analysis)
   ```

## Example

The `review_analyzer.py` file includes a working example in the `main()` function. Run it with:

```bash
python review_analyzer.py
```

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in requirements.txt

## Notes

- The vector store is persisted in the `./chroma_db` directory
- Reviews are automatically split into chunks for better analysis
- The system uses GPT-3.5-turbo for analysis 