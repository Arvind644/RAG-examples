# Document Chatbot

An intelligent chatbot that can answer questions about your documents using OpenAI's language models and FAISS vector store.

## Features

- Support for multiple document types (PDF, TXT, DOCX)
- Interactive terminal-based chat interface
- Persistent vector store for faster subsequent loads
- Configurable settings via config.json
- Progress indicators for long operations
- Chat history management
- Comprehensive error handling and logging

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'  # On Windows: set OPENAI_API_KEY=your-api-key-here
   ```

## Usage

1. Run the chatbot:
   ```bash
   python pdf_chatbot.py
   ```

2. Choose from the following options:
   - Process specific files
   - Process all files in a directory
   - Use existing vector store
   - Exit

3. Chat commands:
   - `/clear` - Clear chat history
   - `/exit` - Exit the program
   - `/help` - Show help message

## Configuration

The `config.json` file (created automatically on first run) contains the following settings:

```json
{
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "temperature": 0,
    "supported_extensions": [".pdf", ".txt", ".docx"],
    "vectorstore_path": "vectorstore",
    "model_name": "gpt-3.5-turbo"
}
```

## Logging

Logs are written to `pdf_chatbot.log` and also displayed in the console.

## Requirements

- Python 3.7+
- OpenAI API key
- See requirements.txt for Python package dependencies 