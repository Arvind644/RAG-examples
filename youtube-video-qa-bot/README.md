# YouTube Video QA Bot

This is a command-line application that allows you to ask questions about YouTube videos. It uses the video's captions to create a knowledge base and answers questions using OpenAI's language models.

## Features

- Fetches captions from YouTube videos
- Creates embeddings using OpenAI's API
- Stores embeddings in ChromaDB for efficient retrieval
- Interactive Q&A interface
- Supports both youtube.com and youtu.be URLs

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the script:
   ```bash
   python youtube_qa_bot.py
   ```
2. Enter a YouTube video URL when prompted
3. Wait for the transcript to be processed
4. Ask questions about the video content
5. Type 'quit' to exit

## Notes

- The video must have captions available
- Processing time depends on the video length
- The first run will create a `chroma_db` directory to store embeddings 