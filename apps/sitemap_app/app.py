import os
import streamlit as st
import logging
import sys
import time
import random
from typing import List, Dict, Any
from datetime import timedelta
from time import sleep

from openai import OpenAI
from openai import RateLimitError, APIError
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import WebBaseLoader, SitemapLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document

from bs4 import BeautifulSoup

# 1) LangChain의 사용자 에이전트 설정
if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("sitemap.log")],
)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(page_title="Cloudflare Docs GPT", page_icon="📚", layout="centered")

###########################################
# 0. 전역 상수 및 초기 설정
###########################################
PRODUCT_INFO = {
    "AI Gateway": {
        "url": "https://developers.cloudflare.com/ai-gateway/",
        "example_questions": [
            "AI Gateway의 기본 가격은 얼마인가요?",
            "llama-2-7b-chat-fp16 모델의 1M 토큰당 가격은?",
            "AI Gateway에서 지원하는 모델 목록은?",
        ],
    },
    "Vectorize": {
        "url": "https://developers.cloudflare.com/vectorize/",
        "example_questions": [
            "Vectorize의 인덱스 제한은 얼마인가요?",
            "단일 계정의 최대 인덱스 수는?",
            "Vectorize의 가격 정책은 어떻게 되나요?",
        ],
    },
    "Workers AI": {
        "url": "https://developers.cloudflare.com/workers-ai/",
        "example_questions": [
            "Workers AI에서 사용 가능한 모델은?",
            "추론 API의 가격은 얼마인가요?",
            "Workers AI의 주요 기능은 무엇인가요?",
        ],
    },
}

SITEMAP_URL = "https://developers.cloudflare.com/sitemap-0.xml"

RELEVANT_KEYWORDS = [
    "price",
    "cost",
    "model",
    "api",
    "limit",
    "quota",
    "feature",
    "token",
    "gateway",
    "vector",
    "worker",
    "ai",
    "inference",
    "embed",
    "index",
    "rate",
    "pricing",
    "free",
    "tier",
    "usage",
]


###########################################
# 1. 헬퍼 함수 및 콜백 클래스
###########################################
class ChatCallbackHandler(BaseCallbackHandler):
    """LLM에서 streaming(True)로 토큰이 들어올 때마다
    Streamlit UI에 한 토큰씩 출력하기 위한 Callback
    """

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message = ""  # Reset message at start
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "assistant")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def format_time(seconds: float) -> str:
    """시/분/초 형태로 표시"""
    return str(timedelta(seconds=int(seconds))).split(".")[0]


def custom_parse_function(soup: BeautifulSoup) -> str:
    """HTML 파싱 함수: 불필요한 섹션은 제거하고 본문만 추출"""
    for element in soup.find_all(["nav", "header", "footer", "script", "style"]):
        element.decompose()
    main_content = soup.find("main") or soup.find("article") or soup.find("body")
    if main_content:
        return main_content.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)


def is_valid_openai_api_key(api_key: str) -> bool:
    """OpenAI API 키 유효성 검증"""
    if not api_key or not api_key.startswith("sk-"):
        return False
    try:
        client = OpenAI(api_key=api_key)
        _ = client.models.list()
        return True
    except Exception as e:
        logger.error(f"API key validation error: {str(e)}")
        return False


def get_accessible_embedding_models(api_key: str) -> List[str]:
    """Accessible embedding models fetched from OpenAI"""
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        accessible_models = [
            model.id for model in models.data if "embedding" in model.id
        ]
        return accessible_models
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        return []


def get_accessible_llm_models(api_key: str) -> List[str]:
    """Accessible LLM models fetched from OpenAI"""
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        accessible_models = [
            model.id
            for model in models.data
            if model.id.startswith("gpt-") or model.id.startswith("o1")
        ]
        return accessible_models
    except Exception as e:
        logger.error(f"Error fetching LLM models: {str(e)}")
        return []


def filter_relevant_docs(docs: List[Document]) -> List[Document]:
    """특정 키워드와 매칭되는 문서만 추출"""
    return [
        doc
        for doc in docs
        if any(keyword in doc.page_content.lower() for keyword in RELEVANT_KEYWORDS)
    ]


###########################################
# 2. 문서 로더 함수
###########################################
def load_docs_from_sitemap(sitemap_url: str) -> List[Dict[str, Any]]:
    """사이트맵에서 문서를 로드하는 함수"""
    try:
        loader = SitemapLoader(
            web_path=sitemap_url,
            filter_urls=[
                r"https://developers\.cloudflare\.com/ai-gateway/.*",
                r"https://developers\.cloudflare\.com/vectorize/.*",
                r"https://developers\.cloudflare\.com/workers-ai/.*",
            ],
            parsing_function=custom_parse_function,
            requests_per_second=2,
            requests_kwargs={
                "headers": {
                    "User-Agent": os.environ["USER_AGENT"],
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                },
            },
        )
        docs = loader.load()
        filtered_docs = filter_relevant_docs(docs)
        logger.info(f"Loaded {len(filtered_docs)} documents from sitemap")
        return filtered_docs
    except Exception as e:
        logger.error(f"Error loading sitemap: {str(e)}", exc_info=True)
        return []


def load_docs_from_url(url: str, max_docs: int = 32) -> List[Dict[str, Any]]:
    """단일 URL에서 문서를 로드하는 함수"""
    try:
        headers = {
            "User-Agent": os.environ["USER_AGENT"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }

        base_loader = WebBaseLoader(
            web_path=[url],
            continue_on_failure=True,
            requests_kwargs={
                "timeout": 60,
                "headers": headers,
                "verify": False,
            },
        )

        try:
            docs = base_loader.load()
            logger.info(f"Loaded {len(docs)} documents from {url}")
        except Exception as e:
            logger.error(f"Error loading base URL {url}: {str(e)}", exc_info=True)
            docs = []

        if docs:
            filtered_docs = filter_relevant_docs(docs)
            final_docs = filtered_docs[:max_docs] if max_docs else filtered_docs
            return final_docs
        return []
    except Exception as e:
        logger.error(f"Error in load_docs_from_url for {url}: {str(e)}", exc_info=True)
        return []


###########################################
# 3. 문서 벡터화 함수
###########################################
def initialize_vectorstore(_docs: List[Dict[str, Any]], api_key: str, embed_model: str):
    """문서를 chunking -> 임베딩 -> FAISS에 저장하는 함수"""
    try:
        if not _docs:
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=32,
            length_function=len,
        )
        splits = text_splitter.split_documents(_docs)

        status_container = st.container()
        with status_container:
            st.markdown("### Processing Documents")
            progress_text = st.empty()
            progress_bar = st.progress(0)
            time_text = st.empty()
            stats_text = st.empty()

            start_time = time.time()
            vectorstore = None
            total_splits = len(splits)
            processed_chunks = 0
            BATCH_SIZE = 32
            total_batches = (total_splits + BATCH_SIZE - 1) // BATCH_SIZE

            try:
                embeddings = OpenAIEmbeddings(
                    api_key=api_key,
                    model=embed_model,
                    chunk_size=32,
                    max_retries=3,
                    timeout=60,
                )
            except APIError as e:
                st.error(f"OpenAI API Error during embeddings initialization: {e}")
                logger.error(
                    f"Embeddings initialization error: {str(e)}", exc_info=True
                )
                return None
            except Exception as e:
                st.error(f"Failed to initialize embeddings: {str(e)}")
                logger.error(
                    f"Embeddings initialization error: {str(e)}", exc_info=True
                )
                return None

            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, total_splits)
                batch = splits[start_idx:end_idx]

                retry_count = 0
                max_retries = 4

                while True:
                    try:
                        temp_vectorstore = FAISS.from_documents(batch, embeddings)
                        if vectorstore is None:
                            vectorstore = temp_vectorstore
                        else:
                            vectorstore.merge_from(temp_vectorstore)
                        break
                    except RateLimitError:
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.error(f"Exceeded max retries for batch {batch_idx}.")
                            st.error("Rate limit exceeded. Please try again later.")
                            return None
                        wait_time = (2**retry_count) + random.random()
                        logger.warning(
                            f"[429] Rate limit. Retrying batch {batch_idx} in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    except APIError as e:
                        logger.error(f"OpenAI API Error: {str(e)}", exc_info=True)
                        st.error(f"OpenAI API Error: {str(e)}")
                        return None
                    except Exception as e:
                        logger.error(
                            f"Error processing batch {batch_idx}: {str(e)}",
                            exc_info=True,
                        )
                        st.error(f"Error processing documents: {str(e)}")
                        return None

                processed_chunks += len(batch)
                progress = (batch_idx + 1) / total_batches
                progress_bar.progress(progress)

                elapsed = time.time() - start_time
                estimated_total = elapsed / progress if progress > 0 else 0
                remaining = estimated_total - elapsed

                progress_text.text(f"Processing batch {batch_idx+1}/{total_batches}")
                time_text.markdown(
                    f"⏱️ **Time Info**\n"
                    f"- Elapsed: {format_time(elapsed)}\n"
                    f"- Remaining: {format_time(remaining)}"
                )
                stats_text.markdown(
                    f"📊 **Processing Stats**\n"
                    f"- Chunks processed: {processed_chunks}/{total_splits}\n"
                    f"- Speed: {processed_chunks/elapsed:.1f} chunks/sec"
                )

                sleep(0.2)

            return vectorstore

    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}", exc_info=True)
        st.error(f"Error initializing vector store: {str(e)}")
        return None


###########################################
# 4. Chat 모델 설정
###########################################
def setup_chat_model(api_key: str, chat_model: str):
    """LLM 초기화. 응답 포맷팅 개선"""
    try:
        return ChatOpenAI(
            api_key=api_key,
            model=chat_model,
            temperature=0.1,
            streaming=True,
            callbacks=[ChatCallbackHandler()],
            timeout=60,
            max_retries=3,
        )
    except Exception as e:
        logger.error(f"Error initializing chat model: {str(e)}", exc_info=True)
        st.error(f"Failed to initialize chat model: {str(e)}")
        return None


def setup_prompt_template() -> str:
    """프롬프트 템플릿"""
    return """You are a helpful assistant that answers questions about Cloudflare's documentation.
You MUST follow these guidelines:

1. Use ONLY the provided context to answer the question
2. If specific pricing, limits, or numerical values are available, always include them
3. If a source URL is provided, include it as a markdown link in your response
4. Keep responses direct and well-structured
5. If information is missing or uncertain, clearly state what is not available
6. Always mention specific products/features by their exact names
7. Format the answer in clear Markdown with appropriate headers and lists if needed
8. For pricing questions, break down the costs clearly if multiple tiers exist

Context:
{context}

Question:
{question}

Guidelines for response format:
1. Start with a direct answer
2. Include specific details and numbers
3. Add source links
4. Explain any limitations or conditions
5. Use proper Markdown formatting"""


def extract_doc_info(doc: Document) -> Dict[str, str]:
    """문서에서 중요 정보 추출"""
    source = doc.metadata.get("source", "N/A")
    if source.startswith("/"):
        source = f"https://developers.cloudflare.com{source}"

    content = doc.page_content

    # 중요 정보 추출
    pricing_info = []
    limit_info = []
    feature_info = []

    for line in content.split("\n"):
        line = line.strip()
        if any(word in line.lower() for word in ["price", "cost", "$", "free"]):
            pricing_info.append(line)
        if any(word in line.lower() for word in ["limit", "quota", "maximum"]):
            limit_info.append(line)
        if any(word in line.lower() for word in ["feature", "support", "provide"]):
            feature_info.append(line)

    return {
        "source": source,
        "content": content,
        "pricing": " ".join(pricing_info),
        "limits": " ".join(limit_info),
        "features": " ".join(feature_info),
    }


def process_context(docs: List[Document]) -> str:
    """문맥 정보 처리 개선"""
    processed_contexts = []

    for doc in docs:
        doc_info = extract_doc_info(doc)
        context_parts = []

        # 기본 정보 추가
        context_parts.append(f"Source: {doc_info['source']}")

        # 중요 정보가 있는 경우 강조
        if doc_info["pricing"]:
            context_parts.append(f"Pricing Information: {doc_info['pricing']}")
        if doc_info["limits"]:
            context_parts.append(f"Limits/Quotas: {doc_info['limits']}")
        if doc_info["features"]:
            context_parts.append(f"Features: {doc_info['features']}")

        # 나머지 컨텐츠
        context_parts.append(f"Additional Content: {doc_info['content']}")

        processed_contexts.append("\n".join(context_parts))

    return "\n\n---\n\n".join(processed_contexts)


def generate_chat_response(llm: ChatOpenAI, docs: List[Document], question: str) -> str:
    """개선된 응답 생성 로직"""
    try:
        context_text = process_context(docs)
        prompt_template = ChatPromptTemplate.from_template(setup_prompt_template())
        chain = prompt_template | llm
        response = chain.invoke({"context": context_text, "question": question})

        # 응답 처리
        if hasattr(response, "content"):
            return response.content
        return str(response)

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        return "Sorry, I encountered an error while generating the response. Please try again."


###########################################
# 5. Streamlit 세션 상태 초기화
###########################################
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False
    if "selected_products" not in st.session_state:
        st.session_state.selected_products = []
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = "text-embedding-3-small"
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = "gpt-4o-mini-2024-07-18"


###########################################
# 6. 채팅 메시지 핸들러
###########################################
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


###########################################
# 7. 문서 로딩 및 VectorStore 초기화
###########################################
def process_document_loading(
    api_key: str, selected_urls: Dict[str, str], embed_model: str
):
    """문서 로드 + 벡터스토어 생성"""
    try:
        start_time = time.time()
        all_docs = []

        with st.spinner("Loading documents from sitemap..."):
            sitemap_docs = load_docs_from_sitemap(SITEMAP_URL)
            if sitemap_docs:
                all_docs.extend(sitemap_docs)
                st.success(f"Loaded {len(sitemap_docs)} documents from sitemap")

        if not all_docs or len(all_docs) < len(selected_urls):
            loading_container = st.container()
            with loading_container:
                st.markdown("### Loading Documentation from URLs")
                progress_text = st.empty()
                progress_bar = st.progress(0)
                time_text = st.empty()
                stats_text = st.empty()

                total_urls = len(selected_urls)
                failed_products = []

                for idx, (product, url) in enumerate(selected_urls.items(), 1):
                    progress_text.text(
                        f"Loading {product} documentation ({idx}/{total_urls})..."
                    )

                    docs = load_docs_from_url(url, max_docs=50)
                    api_url = url.rstrip("/") + "/api/"
                    api_docs = load_docs_from_url(api_url, max_docs=50)

                    if docs or api_docs:
                        all_docs.extend(docs)
                        all_docs.extend(api_docs)
                    else:
                        failed_products.append(product)

                    progress = idx / total_urls
                    progress_bar.progress(progress)

                    elapsed = time.time() - start_time
                    estimated_total = elapsed / progress if progress > 0 else 0
                    remaining = estimated_total - elapsed

                    time_text.markdown(
                        f"""
                        ⏱️ **Time Info**  
                        - Elapsed: {format_time(elapsed)}  
                        - Remaining: {format_time(remaining)}
                        """
                    )
                    stats_text.markdown(
                        f"""
                        📊 **Progress**  
                        - Documents loaded for **{product}**: {len(docs) + len(api_docs)}  
                        - Total documents so far: {len(all_docs)}
                        """
                    )

                if failed_products:
                    st.warning(
                        f"⚠️ Some products failed to load: {', '.join(failed_products)}"
                    )

        if all_docs:
            st.info(f"Processing {len(all_docs)} documents...")
            st.session_state.vectorstore = initialize_vectorstore(
                all_docs, api_key, embed_model
            )
            if st.session_state.vectorstore:
                st.session_state.docs_loaded = True
                st.success("✅ Documentation loaded successfully!")
            else:
                st.error(
                    "Failed to process documents. Please check the logs for more details."
                )
        else:
            st.error("No documentation was loaded. Please try again.")
            return

    except Exception as e:
        st.error(f"Error loading documentation: {str(e)}")
        logger.error(f"Documentation loading error: {str(e)}", exc_info=True)


###########################################
# 8. 사이드바 설정
###########################################
def setup_sidebar():
    with st.sidebar:
        st.title("📚 Cloudflare Docs GPT")

        with st.expander("🔑 API & Model Settings", expanded=True):
            with st.form("api_model_form"):
                st.markdown("**Enter your OpenAI API Key** (format: `sk-...`) :")
                api_key_input = st.text_input(
                    "OpenAI API Key",
                    value=st.session_state.get("openai_api_key", ""),
                    type="password",
                    placeholder="sk-...",
                )

                if is_valid_openai_api_key(api_key_input):
                    accessible_embedding_models = get_accessible_embedding_models(
                        api_key_input
                    )
                    accessible_llm_models = get_accessible_llm_models(api_key_input)
                else:
                    accessible_embedding_models = []
                    accessible_llm_models = []

                embedding_model = st.selectbox(
                    "Select Embedding Model:",
                    options=(
                        accessible_embedding_models
                        if accessible_embedding_models
                        else ["text-embedding-3-small"]
                    ),
                    index=0,
                    help="Pick a model that your project has access to.",
                )

                chat_model = st.selectbox(
                    "Select Chat Model:",
                    options=(
                        accessible_llm_models
                        if accessible_llm_models
                        else ["gpt-4o-mini-2024-07-18"]
                    ),
                    index=0,
                    help="Pick a chat model your project has access to.",
                )

                submitted = st.form_submit_button("Validate & Save")
                if submitted:
                    if is_valid_openai_api_key(api_key_input):
                        st.session_state["openai_api_key"] = api_key_input
                        st.session_state["embedding_model"] = embedding_model
                        st.session_state["chat_model"] = chat_model
                        st.success("✅ API key is valid!")
                    else:
                        st.error("❌ Invalid API Key. Please check and try again.")

        st.markdown("---")

        selected_products = st.multiselect(
            "Choose products:",
            options=list(PRODUCT_INFO.keys()),
            default=["AI Gateway"],
            help="Select one or more products",
        )

        if selected_products:
            st.success(f"Selected: {', '.join(selected_products)}")
            st.session_state["selected_products"] = selected_products

            with st.expander("📝 Example Questions", expanded=False):
                for product in selected_products:
                    st.markdown(f"**{product}**")
                    for question in PRODUCT_INFO[product]["example_questions"]:
                        st.markdown(f"- {question}")

        st.markdown("---")

        st.markdown("### Documentation Status")
        if st.session_state.get("docs_loaded"):
            st.success("Documentation loaded", icon="✅")
            if st.button("🔄 Reload Documentation"):
                st.session_state.docs_loaded = False
                st.session_state.vectorstore = None
                st.session_state.messages = []
                st.info("Documentation state cleared. Please refresh the page.")
        else:
            st.warning("Documentation not loaded", icon="⚠️")
            if st.button("📥 Load Selected Documentation"):
                api_key_now = st.session_state.get("openai_api_key", "")
                if not api_key_now or not is_valid_openai_api_key(api_key_now):
                    st.error("Please provide a valid OpenAI API key.")
                    return None, None
                if not selected_products:
                    st.error("Please select at least one product.")
                    return None, None

                selected_urls = {
                    product: PRODUCT_INFO[product]["url"]
                    for product in selected_products
                }
                process_document_loading(
                    api_key_now, selected_urls, st.session_state.get("embedding_model")
                )

        st.markdown("---")

        if st.button("🗑️ Clear Chat History"):
            st.session_state["messages"] = []
            st.info("Chat history cleared. Please refresh if needed.")

        st.markdown("---")
        st.markdown("### Links")
        st.markdown("- [Cloudflare Developer Docs](https://developers.cloudflare.com/)")
        st.markdown(
            "- [GitHub Repository](https://github.com/researcherhojin/langchain-practice)"
        )

        return st.session_state.get("openai_api_key", ""), selected_products


###########################################
# 9. Main Streamlit App
###########################################
def handle_chat_interaction(prompt: str, api_key: str):
    """채팅 상호작용 처리 로직"""
    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                # VectorStore에서 검색
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr", search_kwargs={"k": 4}
                )
                # invoke가 직접 문서 리스트를 반환
                docs = retriever.invoke(prompt)

                if not docs:
                    st.error("No relevant information found in the documentation.")
                    return

                # LLM 초기화
                llm = setup_chat_model(
                    api_key=api_key,
                    chat_model=st.session_state.get(
                        "chat_model", "gpt-4o-mini-2024-07-18"
                    ),
                )

                if not llm:
                    st.error("Failed to initialize the chat model.")
                    return

                # 응답 생성
                context_text = process_context(docs)
                prompt_template = ChatPromptTemplate.from_template(
                    setup_prompt_template()
                )
                chain = prompt_template | llm
                chain.invoke({"context": context_text, "question": prompt})

    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        logger.error(f"Chat interaction error: {str(e)}", exc_info=True)


def main():
    """Main Streamlit App"""
    try:
        initialize_session_state()
        api_key, selected_products = setup_sidebar()

        st.title("🔍 Cloudflare Docs GPT")

        # 초기 안내 메시지
        if not st.session_state.get("docs_loaded"):
            st.markdown(
                """
                Welcome to **Cloudflare Docs GPT**! 👋

                **How to use**:
                1. Enter your **OpenAI API key** in the sidebar
                2. Select products you want to explore
                3. Click **Load Selected Documentation**
                4. Start asking questions about Cloudflare products!

                **Features**:
                - Get accurate information from official documentation
                - Access pricing, limits, and feature details
                - View relevant documentation sources
                - Clear and structured responses
                """
            )
        else:
            # 선택된 제품 표시
            products_str = ", ".join(st.session_state.get("selected_products", []))
            st.markdown(
                f"""
                **Currently loaded products**: {products_str}

                Ask any questions about:
                - Pricing and costs
                - Usage limits and quotas
                - Features and capabilities
                - API details and endpoints
                - Implementation guides
                """
            )

        # 채팅 기록 표시
        paint_history()

        # 채팅 입력 처리
        prompt = st.chat_input("Ask about Cloudflare documentation...")
        if prompt:
            # 입력 검증
            if not api_key or not is_valid_openai_api_key(api_key):
                st.error("Please provide a valid OpenAI API key in the sidebar.")
                return
            if not st.session_state.get("vectorstore"):
                st.error("Please load the documentation first.")
                return
            if not st.session_state.get("selected_products"):
                st.error("Please select at least one product from the sidebar.")
                return

            # 사용자 메시지 표시
            send_message(prompt, "human")

            # 채팅 처리
            handle_chat_interaction(prompt, api_key)

        # 페이지 하단 정보
        with st.expander("ℹ️ About This App", expanded=False):
            st.markdown(
                """
                This app uses LangChain and OpenAI to provide accurate answers about Cloudflare products.
                It searches through official Cloudflare documentation to find relevant information.
                
                **Note**:
                - Responses are based only on the loaded documentation
                - Make sure to load documentation before asking questions
                - For the most up-to-date information, always verify with official Cloudflare docs
                """
            )

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(
            "An unexpected error occurred. Please try refreshing the page or contact support."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try again.")
