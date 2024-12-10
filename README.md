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

---

## 🛠 Tech Stack

-   Python
-   LangChain
-   OpenAI GPT-4
-   Jupyter Notebook

---

## 📁 Project Structure

```
langchain-practice/
├── README.md
├── .gitignore
├── day01/
│   └── [Generated Poetry Files]
├── day01.ipynb
└── day02.ipynb
```

## 🚀 How to Run

1. **Set Up the Environment**

```bash
# Install required packages
pip install langchain langchain-openai
```

2. Configure Environment Variables

```
# Create a .env file with the following keys
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

3. Run the Notebook

-   Open day01.ipynb in Jupyter Notebook and execute the cells.

## 📌 References

-   [LangChain Official Documentation](https://python.langchain.com/docs/get_started/introduction)
