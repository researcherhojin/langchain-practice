import streamlit as st
import logging
from pathlib import Path
import sys
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
import tempfile
import os
import chardet

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("app.log")],
)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(page_title="Literary Analysis RAG", page_icon="📚", layout="wide")

# 초기 세션 상태 설정
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True, memory_key="history"
    )
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    st.session_state["chain"] = None


# OpenAI API 키 검증 함수
def is_valid_openai_api_key(api_key):
    if not api_key or not isinstance(api_key, str):
        return False
    if not api_key.startswith("sk-") or len(api_key) < 20:
        return False
    return True


# 파일 인코딩 감지 함수
def detect_file_encoding(file_content):
    result = chardet.detect(file_content)
    logger.info(f"Detected encoding: {result}")
    encoding = result["encoding"]
    # 감지 실패 시 기본 utf-8-sig로 설정
    if encoding is None:
        encoding = "utf-8-sig"
    return encoding


# 체인 히스토리 가져오기 함수
def get_chain_history(x):
    if "memory" in st.session_state:
        try:
            return st.session_state["memory"].load_memory_variables({})["history"]
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}")
            return []
    return []


# 다양한 인코딩을 시도하여 문서를 로드하는 함수
def load_documents_with_multiple_encodings(file_path, primary_encoding):
    """주어진 파일을 여러 인코딩을 시도하여 문자열로 로드한 후 Document 객체로 반환"""
    encodings_to_try = [primary_encoding, "cp949", "utf-8-sig", "latin-1"]
    with open(file_path, "rb") as f:
        raw_data = f.read()

    for enc in encodings_to_try:
        try:
            text = raw_data.decode(enc)
            return [Document(page_content=text)]
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode with {enc}, trying next...")

    # 모든 인코딩 시도 실패
    raise RuntimeError(f"Unable to load document with fallback encodings.")


# RAG 파이프라인 설정 함수
def setup_rag_pipeline(file_path, openai_api_key, encoding):
    try:
        logger.info(f"Attempting to load file from path: {file_path}")
        logger.info(f"File exists: {Path(file_path).exists()}")
        logger.info(f"File size: {Path(file_path).stat().st_size} bytes")
        logger.info(f"Using encoding: {encoding}")

        # 문서 로드 (다양한 인코딩 시도)
        documents = load_documents_with_multiple_encodings(file_path, encoding)
        logger.info("Successfully loaded document")

        # 텍스트 분할
        text_splitter = CharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, separator="\n"
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(splits)} chunks")

        # 임베딩 설정
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # 벡터 스토어 생성
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # 프롬프트 템플릿 설정
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "아래의 context를 바탕으로 질문에 정확하고 상세하게 답변해주세요.\n\nContext: {context}",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        # LLM 설정
        model = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            temperature=0.1,
            openai_api_key=openai_api_key,
        )

        # 체인 구성
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
                "history": get_chain_history,
            }
            | prompt
            | model
        )

        return chain

    except Exception as e:
        logger.error(f"Error in setup_rag_pipeline: {str(e)}", exc_info=True)
        raise


# 사이드바 설정
with st.sidebar:
    st.title("📚 Literary Analysis RAG")

    # OpenAI API 키 입력
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Get your API key from https://platform.openai.com/account/api-keys",
    )

    # API 키 상태 표시
    if openai_api_key:
        if is_valid_openai_api_key(openai_api_key):
            st.success("API key provided", icon="✅")
        else:
            st.error("Invalid API key format. It should start with 'sk-'")
            openai_api_key = None
    else:
        st.warning("Please enter your OpenAI API key", icon="⚠️")

    # 깃허브 링크
    st.markdown(
        """
       ### Links
       - [GitHub Repository](https://github.com/researcherhojin/langchain-practice)
       """
    )

    st.markdown("---")

# 메인 페이지 설정
st.title("📚 Literary Analysis RAG")
st.markdown(
    """
   Upload a text file and ask questions about it. 
   The app will use RAG (Retrieval Augmented Generation) to provide accurate answers.
   """
)

# 파일 업로더
uploaded_file = st.file_uploader(
    "Upload a text file", type=["txt"], help="Upload a text file to analyze"
)

# 파일이 업로드되고 API 키가 제공된 경우 처리
if uploaded_file and openai_api_key and is_valid_openai_api_key(openai_api_key):
    try:
        # 파일 내용 읽기
        file_content = uploaded_file.read()
        logger.info(f"Read {len(file_content)} bytes from uploaded file")

        # 파일 인코딩 감지
        detected_encoding = detect_file_encoding(file_content)
        logger.info(f"Detected file encoding: {detected_encoding}")

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", mode="wb"
        ) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
            logger.info(f"Created temporary file at: {tmp_file_path}")

        try:
            # RAG 파이프라인 설정
            st.session_state["chain"] = setup_rag_pipeline(
                tmp_file_path, openai_api_key, detected_encoding
            )

            # 채팅 히스토리 표시
            for message in st.session_state["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # 사용자 입력 처리
            if prompt := st.chat_input("Ask a question about the text"):
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    response = st.session_state["chain"].invoke(prompt)
                    st.markdown(response.content)

                    # 히스토리 업데이트
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response.content}
                    )
                    st.session_state["memory"].save_context(
                        {"input": prompt}, {"output": response.content}
                    )

        except Exception as e:
            logger.error(f"Error during pipeline execution: {str(e)}", exc_info=True)
            st.error(f"An error occurred during processing: {str(e)}")

        finally:
            # 임시 파일 삭제
            try:
                os.unlink(tmp_file_path)
                logger.info(f"Successfully removed temporary file: {tmp_file_path}")
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}", exc_info=True)

    except Exception as e:
        logger.error(f"Error in file upload handling: {str(e)}", exc_info=True)
        st.error(f"An error occurred while handling the uploaded file: {str(e)}")

else:
    if not uploaded_file:
        st.info("Please upload a text file to start.")
    if not openai_api_key or not is_valid_openai_api_key(openai_api_key):
        st.info("Please provide a valid OpenAI API key.")

# 채팅 히스토리 초기화 버튼
if st.sidebar.button("Clear Chat History"):
    st.session_state["messages"] = []
    st.session_state["memory"].clear()
    st.session_state["chain"] = None
    st.experimental_rerun()
