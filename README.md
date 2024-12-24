# LangChain Practice: 21-Day Hands-On Project

## 📝 Introduction

This project is a 21-day hands-on exploration of LangChain's diverse functionalities. Each day focuses on a unique topic, with outputs and insights recorded throughout the project.

-   **Duration:** December 9, 2024 – December 29, 2024
-   **Objective:** Experiment with LangChain's features and build practical applications.

---

## 📅 Progress

### Day 01 (2024.12.09) - Programming Language Poetry Generator

-   Utilized `ChatPromptTemplate` for prompt engineering.
-   Created chains with `LangChain Expression Language (LCEL)`.
-   Generated poetry reflecting characteristics of major programming languages (Python, Java, JavaScript, etc.).
-   Added literary analysis for each generated poem.
-   Implemented automatic saving of outputs.

#### Key Features Used

-   **`ChatPromptTemplate`**: Structured prompt generation.
-   **`ChatOpenAI`**: Interface with GPT models.
-   **`RunnablePassthrough`**: Chain data flow control.
-   **`StreamingStdOutCallbackHandler`**: Real-time output streaming.

#### Result

-   Poetry generation capturing unique traits of programming languages.
-   Professional analysis of generated poetry.
-   Automatic file saving and management system.

### Day 02 (2024.12.10) - Movie Information Chain with Few-Shot Learning

-   Implemented Few-Shot learning using `FewShotChatMessagePromptTemplate`
-   Created structured movie information responses using example-based learning
-   Generated comprehensive movie information including director, cast, budget, revenue, genre, and synopsis
-   Maintained consistent output format through example-based prompting
-   Enhanced response accuracy with temperature control

#### Key Features Used

-   **`FewShotChatMessagePromptTemplate`**: Example-based prompt structuring
-   **`ChatPromptTemplate`**: System and user message formatting
-   **`ChatOpenAI`**: GPT model interface with streaming
-   **Temperature Control**: Maintaining response consistency

#### Result

-   Structured movie information retrieval system
-   Consistent output formatting across different queries
-   Comprehensive movie details in a standardized format

### Day 03 ~ 04 (2024.12.11 ~ 12) - Enhanced Movie Emoji Generator with Memory

-   Implemented advanced memory management using `ConversationBufferMemory`
-   Created custom `EnhancedMovieMemory` class for structured data storage
-   Developed few-shot learning system for consistent emoji generation
-   Added comprehensive search and retrieval functionalities
-   Implemented persistent storage with JSON file backup
-   Added statistical analysis and multi-genre search capabilities
-   Implemented time-based filtering and movie information management

#### Key Features Used

-   **`ConversationBufferMemory`**: Enhanced conversation history tracking
-   **`FewShotChatMessagePromptTemplate`**: Example-based emoji generation
-   **`RunnablePassthrough`**: Advanced chain data flow management
-   **Custom Memory Management**:
    -   Timestamp and genre-based filtering
    -   Persistent storage system
    -   Statistical analysis
    -   Multi-genre search capability

#### Result

-   Movie emoji generation with consistent format
-   Advanced memory system with search capabilities
-   Genre-based and time-based filtering
-   Comprehensive movie information tracking
-   Statistical analysis of movie data

### Day 05~07 (2024.12.13 ~ 15) - Literary Analysis RAG with Memory

-   Implemented RAG (Retrieval Augmented Generation) pipeline for literary text analysis
-   Enhanced text processing with sophisticated chunking strategies
-   Integrated conversation memory for context-aware responses
-   Developed precise retrieval system for literary context
-   Created detailed prompt engineering for accurate text analysis

#### Key Features Used

-   **`TextLoader & CharacterTextSplitter`**: Advanced text processing
-   **`CacheBackedEmbeddings`**: Efficient embedding management
-   **`FAISS`**: Sophisticated vector similarity search
-   **`ConversationBufferMemory`**: Context-aware conversation handling
-   **Chunking Strategy**: Optimized text segmentation for literary analysis

#### Result

-   Accurate literary text analysis system
-   Context-aware response generation
-   Detailed character and plot analysis capabilities
-   Memory-enhanced conversation flow
-   Efficient text retrieval and analysis

### Day 09~11 (2024.12.17 ~ 19) - Streamlit RAG Application with Token Tracking

-   Migrated RAG pipeline to Streamlit web application
-   Implemented automatic file encoding detection
-   Added token usage tracking and cost estimation
-   Developed real-time conversation logging system

#### Key Features Used

-   **`Streamlit`**: Interactive web interface
-   **`TokenUsageTracker`**: Custom token and cost tracking
-   **`Chardet`**: Automatic encoding detection
-   **`FAISS & OpenAIEmbeddings`**: Efficient vector search

#### Result

-   Interactive web-based RAG system
-   Automatic file encoding handling
-   Real-time token usage monitoring
-   Cost-effective API usage tracking
-   Robust error handling and logging

### Day 12~14 (2024.12.20 ~ 22) - QuizGPT with Function Calling

-   Developed interactive quiz generation application using Streamlit
-   Implemented OpenAI function calling for structured quiz format
-   Added difficulty customization and dynamic scoring system
-   Created comprehensive feedback system with explanations
-   Integrated session state management for quiz progression

#### Key Features Used

-   **`Function Calling`**: Structured quiz generation
-   **`StreamLit`**: Interactive web interface
-   **`Session State`**: Quiz progression management
-   **Logging System**: Comprehensive error tracking
-   **Custom Scoring**: Validated score calculation

#### Result

-   Interactive quiz generation system
-   Customizable difficulty levels
-   Real-time feedback and explanations
-   Progress tracking and statistics
-   Retake functionality for practice

### Day 15~16 (2024.12.23 ~ 24) - Cloudflare Docs GPT with LangChain

-   Developed a documentation chatbot for Cloudflare products using LangChain and Streamlit
-   Implemented intelligent document retrieval using FAISS vector store
-   Created streaming response generation with real-time token display
-   Added support for multiple Cloudflare product documentation
-   Implemented embeddings and chat model selection with OpenAI API

#### Key Features Used

-   **`SitemapLoader & WebBaseLoader`**: Efficient documentation loading
-   **`FAISS`**: Vector similarity search for relevant content
-   **`OpenAIEmbeddings`**: Document vectorization
-   **`ChatOpenAI`**: Streaming response generation
-   **`StreamingCallbackHandler`**: Real-time response display
-   **Document Processing**: Chunking and context optimization
-   **Error Handling**: Comprehensive error management

#### Result

-   Interactive documentation chatbot
-   Multi-product documentation support
-   Real-time streaming responses
-   Context-aware answer generation
-   Efficient document retrieval system

---

## 🛠 Tech Stack

-   Python
-   LangChain
-   OpenAI GPT-4
-   Streamlit
-   FAISS
-   Chardet

---

## 📁 Project Structure

```
langchain-practice/
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
├── .env                          # Environment variables
├── .gitignore                    # git-ignored
│
├── apps/                         # Streamlit web applications
│   ├── quiz_app/                 # QuizGPT application
│   │   └── app.py
│   ├── rag_app/                  # RAG application
│   │   └── app.py
│   └── sitemap_app/              # Cloudflare documentation chatbot
│       └── app.py
│
├── notebooks/                    # Jupyter notebooks for daily experiments
│   ├── data/                     # Supporting data files
│   └── day*.ipynb               # Daily experiment notebooks
│
└── results/                      # Generated outputs
    ├── day01/                    # Poetry generation results
    └── movie_memory.json         # Persistent movie data storage
```

---

## 🚀 How to Run

1. **Set Up the Environment**

    ```bash
    # Clone the repository
    git clone https://github.com/researcherhojin/langchain-practice.git
    cd langchain-practice

    # Install required packages
    pip install -r requirements.txt
    ```

2. **Configure Environment Variables**

    ```bash
    # Create .env file in the project root
    OPENAI_API_KEY=your_openai_api_key_here
    ```

3. **Run Applications**

-   For Notebooks:

    ```bash
    jupyter notebook notebooks/
    ```

-   For RAG Application:

    ```bash
    streamlit run apps/rag_app/app.py
    ```

-   For QuizGPT Application:

    ```bash
    streamlit run apps/quiz_app/app.py
    ```

-   For Sitemap Documentation Chatbot:

    ```bash
    streamlit run apps/sitemap_app/app.py
    ```

## 📌 References

-   [LangChain Official Documentation](https://python.langchain.com/docs/get_started/introduction)
