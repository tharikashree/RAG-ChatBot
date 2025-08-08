
### 📄🔍 Gemini RAG Chatbot

This is a Streamlit-based chatbot that uses a **Retrieval-Augmented Generation (RAG)** pipeline to answer questions from a collection of user-uploaded documents. It is built with Google's Gemini API for embeddings and generation, and the FAISS library for an efficient vector store.

-----

### ✨ Features

  * **Multi-Document Support**: Process and chat with various file types, including PDF, TXT, CSV, and Markdown.
  * **Google Gemini Integration**: Uses Google's state-of-the-art Gemini Pro model for powerful and accurate responses.
  * **Efficient Vector Search**: Employs **FAISS** to create and search a vector store for fast and relevant document retrieval.
  * **Streamlit UI**: Provides an intuitive and user-friendly web interface for document uploads and chat.

-----

### 🚀 Getting Started

#### Prerequisites

  * Python 3.8 or higher.
  * A **Google API Key**. You can get one from the [Google AI for Developers](https://www.google.com/search?q=https://ai.google.dev/docs/api_key) website.

#### Installation

1.  **Create a virtual environment** and activate it:

    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

2.  **Install the required packages** using the `requirements.txt` file from the previous response:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your API Key**. Create a file named `.env` in the root directory of your project and add your Google API key:

    ```
    GOOGLE_API_KEY="your-api-key-here"
    ```

-----

### 🖥️ How to Run

1.  Make sure your virtual environment is active.
2.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run your_app_file_name.py
    ```
    (Replace `your_app_file_name.py` with the name of your Python script).
3.  The application will open in your default web browser.

-----

### 🤖 How to Use

1.  **Upload Documents**: Use the "Upload documents" section to select one or more PDF, TXT, CSV, or Markdown files.
2.  **Ask Questions**: Once the documents are uploaded and processed, a chat interface will appear.
3.  **Chat**: Type your question in the input box and press Enter to get a response based on the content of your documents.

-----

### 🛠️ Technologies

  * **Streamlit**: For the web application UI.
  * **LangChain**: The framework for building the RAG pipeline.
  * **Google Generative AI**: Provides the embedding and LLM models.
  * **FAISS**: The vector store for document indexing and retrieval.
  * **PyPDF2, Unstructured**: Libraries used by LangChain for document loading.
  * **python-dotenv**: To manage environment variables.
