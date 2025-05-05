# AI Powered Skill Finder

**AI Powered Skill Finder** is an intelligent tool that helps users discover relevant skills for various job roles using AI. By leveraging advanced **Google Gemini embeddings** and **ChromaDB** for efficient data retrieval, this tool analyzes job descriptions and provides a curated list of essential skills. Built with **Streamlit**, it offers an interactive interface for easy querying and quick results.

## Project Setup

### Requirements

- Python 3.7+
- Streamlit
- ChromaDB
- Google Cloud SDK
- `chromadb`, `streamlit`, `google-genai`, and other necessary libraries (listed in `requirements.txt`).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zaheer6034/skill2genai/
2. Move into the directory
   ```bash
   cd skill2genai
3. Install required libraries
    ```bash
    pip install -r requirements.txt
4. Set up your Google Cloud project and authenticate for accessing the Google Gemini API.

5. Run the app:
     ```bash
      streamlit run app.py
