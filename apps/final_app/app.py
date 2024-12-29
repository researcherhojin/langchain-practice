import streamlit as st
import os
import logging
import time
import threading
from datetime import timedelta
from typing import Optional, Dict, Any, List
from io import BytesIO
import sys
import re
import random
import json
import hashlib
from functools import wraps

from openai import OpenAI
from openai import RateLimitError, APIError
import PyPDF2
import arxiv
import requests
from langchain.callbacks.base import BaseCallbackHandler

# Constants
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
ALLOWED_FILE_TYPES = ["application/pdf"]
SESSION_TIMEOUT = 3600  # 1 hour
MAX_API_VALIDATION_ATTEMPTS = 3
API_VALIDATION_COOLDOWN = 5  # seconds
PDF_MAX_PAGES = 500
ARXIV_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Assistant Instructions
ASSISTANT_INSTRUCTIONS = """
당신은 전문 연구 논문 분석가이며 AI/ML 분야의 교수입니다. 
학생이나 이 분야를 처음 접하는 연구자들에게 논문을 설명해주는 역할을 합니다.

분석 및 설명 방식:

1. 상세한 논문 분석:
   - 논문의 핵심 내용을 명확하고 이해하기 쉽게 설명
   - 기술적 용어나 개념에 대한 자세한 부연 설명 제공
   - 실제 적용 사례나 예시를 통한 이해도 향상

2. 섹션별 분석 제공:
   - Abstract: 연구의 핵심 내용과 중요성을 일반적인 맥락에서 설명
   - Introduction: 연구 배경과 동기를 쉽게 이해할 수 있도록 설명
   - Methodology: 제안된 방법을 단계별로 상세히 설명
   - Results: 실험 결과의 의미와 중요성을 명확히 해석
   - Discussion/Conclusion: 연구의 시사점과 향후 발전 방향 제시

3. 한국어 지원:
   - 필요한 경우 영문 내용의 한국어 번역 제공
   - 전문 용어에 대한 한영 병기
   - 문맥을 고려한 자연스러운 번역

4. 설명 방식:
   - 복잡한 개념을 단계적으로 설명
   - 실제 예시나 비유를 통한 이해도 증진
   - 관련 연구나 배경 지식 보충 설명
   - 필요시 시각화나 도표를 통한 설명

답변은 항상:
- 학술적으로 정확하고
- 이해하기 쉽게 설명되며
- 구체적인 예시를 포함하고
- 건설적인 제안을 제공해야 합니다.

질문자의 이해도를 고려하여 설명의 깊이와 수준을 조절하되, 
항상 정확하고 전문적인 내용을 유지하면서 설명해야 합니다.

추가로 모든 답변은 한국어로 제공하되, 필요한 경우 영문 용어를 병기합니다.
"""

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("paper_analysis.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """시/분/초 형태로 표시"""
    return str(timedelta(seconds=int(seconds))).split(".")[0]


def secure_function(func):
    """보안 검사를 위한 데코레이터"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # 세션 타임아웃 체크
            if "last_activity" in st.session_state:
                if time.time() - st.session_state.last_activity > SESSION_TIMEOUT:
                    clear_session_state()
                    st.error("세션이 만료되었습니다. 다시 로그인해주세요.")
                    return None

            # API 키 검증 상태 확인
            if not st.session_state.get("api_key_verified", False):
                st.warning("API 키 인증이 필요합니다.")
                return None

            result = func(*args, **kwargs)

            # 마지막 활동 시간 업데이트
            st.session_state.last_activity = time.time()

            return result

        except Exception as e:
            handle_error(e, f"Error in {func.__name__}")
            return None

    return wrapper


def clear_session_state():
    """세션 상태 안전하게 초기화"""
    preserved_keys = {"app_init_timestamp", "deployment_mode"}
    for key in list(st.session_state.keys()):
        if key not in preserved_keys:
            del st.session_state[key]


def sanitize_input(text: str) -> str:
    """사용자 입력 살균"""
    if not isinstance(text, str):
        return ""

    # 위험한 문자 제거
    text = re.sub(r'[<>&;"\']', "", text)
    # XSS 방지
    text = text.replace("javascript:", "")
    text = text.replace("data:", "")
    return text.strip()


def hash_file_content(content: bytes) -> str:
    """파일 컨텐츠의 해시값 생성"""
    return hashlib.sha256(content).hexdigest()


def get_initial_analysis(title: str, type: str = "pdf") -> str:
    return f"""I've successfully processed the {type.upper()} paper "{title}". 
    
    I can help you understand this paper by:
    1. 📝 Providing a detailed summary
    2. 🔍 Explaining specific sections
    3. 🌐 Translating content to Korean
    4. 💡 Clarifying technical concepts
    5. 📊 Analyzing the methodology and results
    6. 🔄 Comparing with other research
    
    What aspects of the paper would you like me to explain? You can:
    - Ask for a general overview
    - Focus on specific sections
    - Request Korean translations
    - Get clarification on technical terms
    - Understand the impact and implications
    
    I'll explain everything in detail, making sure to break down complex concepts into understandable terms."""


def handle_error(error: Exception, context: str = ""):
    """Error handling with improved user feedback"""
    error_msg = str(error)
    error_msg = re.sub(r"sk-[a-zA-Z0-9-]+", "[FILTERED]", error_msg)

    logger.error(f"{context} error: {error_msg}", exc_info=True)

    if isinstance(error, PDFExtractionError):
        st.error(
            """
            ### 📄 PDF 처리 오류
            파일을 처리하는 중 문제가 발생했습니다:
            - 파일이 손상되었거나 암호화되어 있을 수 있습니다
            - 지원되지 않는 PDF 형식일 수 있습니다
            - 텍스트 추출이 불가능한 형식일 수 있습니다
            
            다른 PDF 파일을 시도해보시거나, arXiv ID를 사용해보세요.
        """
        )
    elif isinstance(error, ArxivError):
        st.error(
            """
            ### 📚 arXiv 논문 로딩 오류
            arXiv 논문을 불러오는 중 문제가 발생했습니다:
            - arXiv ID가 올바른지 확인해주세요
            - 잠시 후 다시 시도해주세요
            - 다른 방법(PDF 업로드)을 시도해보세요
        """
        )
    elif isinstance(error, AssistantError):
        st.error(
            """
            ### 🤖 AI 분석 오류
            분석 중 문제가 발생했습니다:
            - 잠시 후 다시 시도해주세요
            - 질문을 다른 방식으로 작성해보세요
            - 작은 단위로 나누어 질문해보세요
        """
        )
    else:
        st.error(f"예기치 않은 오류가 발생했습니다: {error_msg}")

    if context:
        st.info("💡 문제가 지속되면 GitHub Issues를 통해 문의해주세요.")


# Exception Classes and Callback Handler
def extract_arxiv_id(url_or_id: str) -> Optional[str]:
    """Extract and validate arXiv ID with improved security"""
    if not url_or_id or not isinstance(url_or_id, str):
        return None

    # Clean input
    cleaned_input = sanitize_input(url_or_id.strip())
    if not cleaned_input:
        return None

    # arXiv ID 패턴
    patterns = [
        r"(?:arxiv.org/(?:abs|pdf)/)?([0-9]{4}\.[0-9]{4,5})",  # 새로운 형식 (YYMM.NNNNN)
        r"(?:arxiv.org/(?:abs|pdf)/)?([a-z-]+/[0-9]{7})",  # 이전 형식 (subject/NNNNNNN)
        r"^([0-9]{4}\.[0-9]{4,5})$",  # 직접 ID 입력 (새로운 형식)
        r"^([a-z-]+/[0-9]{7})$",  # 직접 ID 입력 (이전 형식)
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned_input)
        if match:
            arxiv_id = match.group(1)
            # Additional validation
            if len(arxiv_id) > 20:  # Maximum length check
                logger.warning(f"arXiv ID too long: {arxiv_id}")
                return None

            if re.search(r"[^a-zA-Z0-9\-\./]", arxiv_id):  # Special character check
                logger.warning(f"Invalid characters in arXiv ID: {arxiv_id}")
                return None

            return arxiv_id

    logger.warning(f"No valid arXiv ID found in input: {cleaned_input}")
    return None


class ChatCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming responses with improved error handling"""

    def __init__(self):
        self.message = ""
        self.message_box = None
        self.error_count = 0
        self.MAX_ERRORS = 3
        self._lock = threading.Lock()

    def on_llm_start(self, *args, **kwargs):
        with self._lock:
            self.message = ""
            self.message_box = st.empty()
            self.error_count = 0

    def on_llm_end(self, *args, **kwargs):
        try:
            with self._lock:
                if self.message:
                    save_message(self.message, "assistant")
        except Exception as e:
            logger.error(f"Error in on_llm_end: {str(e)}")

    def on_llm_error(self, error: Exception, *args, **kwargs):
        with self._lock:
            self.error_count += 1
            logger.error(f"LLM error: {str(error)}")
            if self.error_count >= self.MAX_ERRORS:
                raise Exception("Too many LLM errors")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        try:
            with self._lock:
                self.message += token
                sanitized_message = sanitize_input(self.message)
                self.message_box.markdown(sanitized_message)
        except Exception as e:
            logger.error(f"Error in token processing: {str(e)}")
            self.on_llm_error(e)


class PaperAnalysisError(Exception):
    """Base exception for paper analysis with improved error details"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = sanitize_input(message)
        self.details = {k: sanitize_input(str(v)) for k, v in (details or {}).items()}
        super().__init__(self.message)
        logger.error(f"{self.message} - Details: {self.details}")


class PDFExtractionError(PaperAnalysisError):
    """Exception for PDF extraction with validation"""

    def __init__(
        self,
        message: str,
        file_name: Optional[str] = None,
        page_number: Optional[int] = None,
    ):
        details = {
            "file_name": file_name if file_name else "unknown",
            "page_number": page_number if page_number else "unknown",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        super().__init__(message, details)


class ArxivError(PaperAnalysisError):
    """Exception for arXiv operations with validation"""

    def __init__(
        self,
        message: str,
        arxiv_id: Optional[str] = None,
        error_type: Optional[str] = None,
    ):
        details = {
            "arxiv_id": arxiv_id if arxiv_id else "unknown",
            "error_type": error_type if error_type else "unknown",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        super().__init__(message, details)


class AssistantError(PaperAnalysisError):
    """Exception for Assistant API operations with validation"""

    def __init__(
        self,
        message: str,
        assistant_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ):
        details = {
            "assistant_id": assistant_id if assistant_id else "unknown",
            "thread_id": thread_id if thread_id else "unknown",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        super().__init__(message, details)


# Message and State Management
def save_message(message: str, role: str):
    """Save message to session state with validation"""
    if not isinstance(message, str) or not isinstance(role, str):
        logger.error("Invalid message or role type")
        return

    if role not in ["user", "assistant"]:
        logger.error(f"Invalid role: {role}")
        return

    message = sanitize_input(message)
    if not message:
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages.append(
        {
            "role": role,
            "content": message,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


def init_session_state():
    """Initialize session state with improved security"""
    default_states = {
        "openai_client": None,
        "assistant_id": None,
        "thread_id": None,
        "messages": [],
        "current_paper": None,
        "paper_metadata": None,
        "api_key_verified": False,
        "active_tab": "PDF Upload",
        "show_suggestions": False,
        "current_question": None,
        "paper_content": None,
        "processing_status": None,
        "start_time": time.time(),
        "last_activity": time.time(),
        "api_validation_attempts": 0,
        "processed_files": set(),
        "chat_callback_handler": ChatCallbackHandler(),
    }

    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# File Processing and Validation Functions


def validate_pdf_file(uploaded_file) -> bool:
    """PDF 파일 유효성 검증"""
    if not uploaded_file:
        return False

    try:
        # 파일 크기 검증
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(
                f"파일 크기가 너무 큽니다. 최대 {MAX_FILE_SIZE/(1024*1024):.0f}MB까지 가능합니다."
            )
            return False

        # 파일 타입 검증
        if uploaded_file.type not in ALLOWED_FILE_TYPES:
            st.error("PDF 파일만 업로드 가능합니다.")
            return False

        # 파일 내용 검증
        try:
            content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer

            # PDF 파일 형식 검증
            try:
                PyPDF2.PdfReader(BytesIO(content))
            except Exception as e:
                st.error("유효하지 않은 PDF 파일입니다.")
                logger.error(f"Invalid PDF format: {str(e)}")
                return False

            # 해시값 생성 및 저장
            file_hash = hash_file_content(content)
            if "processed_files" not in st.session_state:
                st.session_state.processed_files = set()

            # 중복 파일 체크
            if file_hash in st.session_state.processed_files:
                st.warning("이미 처리된 파일입니다.")
                return False

            st.session_state.processed_files.add(file_hash)
            return True

        except Exception as e:
            logger.error(f"File content validation error: {str(e)}")
            st.error("파일 내용을 검증하는 중 오류가 발생했습니다.")
            return False

    except Exception as e:
        logger.error(f"File validation error: {str(e)}")
        return False


def is_valid_openai_api_key(api_key: str) -> bool:
    """OpenAI API 키 유효성 검증 및 보안 강화"""
    if not api_key or not isinstance(api_key, str):
        return False

    # Basic security checks
    if not api_key.startswith("sk-") or len(api_key) < 40:
        return False

    try:
        # API 키 검증 시도 횟수 제한
        validation_attempts = st.session_state.get("api_validation_attempts", 0)

        if validation_attempts >= MAX_API_VALIDATION_ATTEMPTS:
            time.sleep(API_VALIDATION_COOLDOWN)
            st.session_state.api_validation_attempts = 0

        st.session_state.api_validation_attempts = validation_attempts + 1

        client = OpenAI(api_key=api_key)
        models = client.models.list()

        # 성공 시 시도 횟수 초기화
        st.session_state.api_validation_attempts = 0
        return True

    except Exception as e:
        logger.error(f"API key validation error: {str(e)}")
        return False


@secure_function
def extract_text_from_pdf(
    file_content: BytesIO, file_name: Optional[str] = None
) -> str:
    """Extract text from PDF with enhanced security and validation"""
    try:
        reader = PyPDF2.PdfReader(file_content)

        # PDF 페이지 수 제한 확인
        total_pages = len(reader.pages)
        if total_pages > PDF_MAX_PAGES:
            raise PDFExtractionError(
                f"PDF page limit exceeded: {total_pages} > {PDF_MAX_PAGES}",
                file_name=file_name,
            )

        text = []
        progress_text = st.empty()
        progress_bar = st.progress(0)

        for i, page in enumerate(reader.pages):
            try:
                progress_text.text(f"Processing page {i+1} of {total_pages}")
                progress_bar.progress((i + 1) / total_pages)

                page_text = page.extract_text()
                if not page_text:
                    logger.warning(f"Empty text on page {i+1}")
                    continue

                text.append(page_text)

            except Exception as e:
                raise PDFExtractionError(
                    f"Error on page {i+1}: {str(e)}",
                    file_name=file_name,
                    page_number=i + 1,
                )

        progress_text.empty()
        progress_bar.empty()
        return "\n".join(text)

    except PDFExtractionError:
        raise
    except Exception as e:
        raise PDFExtractionError(f"Error processing PDF: {str(e)}", file_name=file_name)


@secure_function
def load_from_arxiv(arxiv_id: str) -> Dict[str, Any]:
    """Load paper from arXiv with enhanced security and validation"""
    try:
        if hasattr(st.session_state, "processing_status"):
            st.session_state.processing_status = {
                "message": "Fetching from arXiv...",
                "progress": 0.2,
                "elapsed": format_time(time.time() - st.session_state.start_time),
                "details": f"Retrieving arXiv ID: {arxiv_id}",
            }

        # arXiv ID 검증
        if not re.match(r"^[\d.]+$|^[\w-]+/\d{7}$", arxiv_id):
            raise ArxivError("Invalid arXiv ID format", arxiv_id=arxiv_id)

        # API 요청 타임아웃 설정
        search = arxiv.Search(
            id_list=[arxiv_id], max_results=1, sort_by=arxiv.SortCriterion.SubmittedDate
        )

        try:
            paper = next(search.results())
        except StopIteration:
            raise ArxivError("Paper not found", arxiv_id=arxiv_id)

        # PDF 다운로드
        if hasattr(st.session_state, "processing_status"):
            st.session_state.processing_status.update(
                {"progress": 0.4, "message": "Downloading PDF..."}
            )

        response = requests.get(
            paper.pdf_url, timeout=ARXIV_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"}
        )

        if response.status_code != 200:
            raise ArxivError(
                "Failed to download PDF",
                arxiv_id=arxiv_id,
                error_type="download_failed",
            )

        pdf_content = BytesIO(response.content)

        # 텍스트 추출
        if hasattr(st.session_state, "processing_status"):
            st.session_state.processing_status.update(
                {"progress": 0.6, "message": "Extracting text..."}
            )

        text = extract_text_from_pdf(pdf_content, f"arxiv_{arxiv_id}.pdf")

        if hasattr(st.session_state, "processing_status"):
            st.session_state.processing_status.update(
                {"progress": 1.0, "message": "Processing complete!"}
            )

        return {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.summary,
            "full_text": text,
            "published": paper.published.strftime("%Y-%m-%d"),
            "categories": paper.categories,
            "doi": paper.doi,
            "arxiv_url": paper.pdf_url,
        }

    except ArxivError:
        raise
    except Exception as e:
        raise ArxivError(
            f"Error processing arXiv paper: {str(e)}",
            arxiv_id=arxiv_id,
            error_type="processing_failed",
        )


def create_assistant(client: OpenAI) -> str:
    """Create or load OpenAI Assistant with improved error handling"""
    try:
        assistant = client.beta.assistants.create(
            name="Research Paper Analysis Assistant",
            instructions=ASSISTANT_INSTRUCTIONS,
            model="gpt-4o-mini-2024-07-18",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_paper",
                        "description": "Analyze academic paper from PDF or arXiv",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "source_type": {
                                    "type": "string",
                                    "enum": ["pdf", "arxiv"],
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Paper content or arXiv ID",
                                },
                            },
                            "required": ["source_type", "content"],
                        },
                    },
                }
            ],
        )
        logger.info(f"Assistant created with ID: {assistant.id}")
        return assistant.id
    except Exception as e:
        logger.error(f"Failed to create assistant: {e}")
        raise AssistantError(f"Error creating assistant: {str(e)}")


@secure_function
def process_message(
    client: OpenAI, thread_id: str, assistant_id: str, message: str
) -> str:
    """Process message with improved error handling and streaming responses"""
    try:
        # 입력 메시지 검증
        message = sanitize_input(message)
        if not message:
            raise ValueError("Empty or invalid message")

        # 현재 문서 내용 가져오기
        paper_content = st.session_state.paper_content
        if not paper_content:
            raise AssistantError("Paper content not available")

        # 메시지에 문서 컨텍스트 추가
        context_message = f"""Based on the following paper content, please answer the question: "{message}"

Paper content:
{paper_content[:10000]}  # First 10000 characters for context

Question: {message}
"""

        # 메시지 생성
        client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=context_message
        )

        # 실행 생성
        run = client.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=assistant_id
        )

        retry_count = 0
        with st.status("Analysis in progress...", expanded=True) as status:
            while True:
                try:
                    run_status = client.beta.threads.runs.retrieve(
                        thread_id=thread_id, run_id=run.id
                    )

                    if run_status.status == "completed":
                        status.update(label="✅ Analysis complete!", state="complete")
                        break
                    elif run_status.status == "failed":
                        raise AssistantError("Analysis failed")
                    elif run_status.status == "requires_action":
                        tool_calls = (
                            run_status.required_action.submit_tool_outputs.tool_calls
                        )
                        tool_outputs = []

                        for tool_call in tool_calls:
                            if tool_call.function.name == "analyze_paper":
                                arguments = json.loads(tool_call.function.arguments)
                                status.update(
                                    label=f"Processing {arguments['source_type']}..."
                                )

                                tool_outputs.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "output": json.dumps(
                                            {
                                                "content": paper_content,
                                                "metadata": st.session_state.paper_metadata,
                                            }
                                        ),
                                    }
                                )

                        client.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread_id,
                            run_id=run.id,
                            tool_outputs=tool_outputs,
                        )

                    status.update(
                        label=f"Analysis in progress... ({run_status.status})"
                    )
                    time.sleep(1)

                except RateLimitError:
                    retry_count += 1
                    if retry_count > MAX_RETRIES:
                        raise
                    time.sleep(RETRY_DELAY * retry_count)
                    continue

        # 응답 검색
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        if not messages.data:
            raise AssistantError("No response received")

        return messages.data[0].content[0].text.value

    except Exception as e:
        logger.error(f"Message processing failed: {e}")
        raise AssistantError(f"Error processing message: {str(e)}")


@secure_function
def handle_paper_upload(uploaded_file):
    """Handle PDF file upload with improved processing status"""
    try:
        if not validate_pdf_file(uploaded_file):
            return False

        with st.spinner("🔄 Initializing paper processing..."):
            st.session_state.start_time = time.time()
            paper_content = uploaded_file.read()

        # Processing status container
        status_container = st.empty()
        progress_container = st.empty()
        step_container = st.empty()

        with status_container.container():
            st.info("📄 Processing PDF document...")

            # Text extraction
            with progress_container:
                text = extract_text_from_pdf(BytesIO(paper_content), uploaded_file.name)

            step_container.success("✅ Text extraction complete")

            # Store paper information
            st.session_state.current_paper = paper_content
            st.session_state.paper_metadata = {
                "type": "pdf",
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.paper_content = text

            # Final success message
            status_container.success(
                f"""
                ### ✅ Document Processing Complete
                - 📄 Filename: {uploaded_file.name}
                - 📏 Size: {uploaded_file.size / 1024:.1f} KB
                - ⌛ Processing Time: {format_time(time.time() - st.session_state.start_time)}
                
                You can now ask questions about the paper!
            """
            )

            # Add initial analysis message
            welcome_msg = f"""I've successfully processed the paper "{uploaded_file.name}". 
            You can ask me to:
            - Summarize the entire paper
            - Explain specific sections
            - Translate parts of the paper
            - Clarify technical concepts
            - Compare with other research
            
            What would you like to know about the paper?"""

            st.session_state.messages.append(
                {"role": "assistant", "content": welcome_msg}
            )

        return True

    except Exception as e:
        handle_error(e, "Paper upload")
        return False


@secure_function
def handle_arxiv_input(arxiv_input: str):
    """Handle arXiv ID input with improved processing status"""
    try:
        arxiv_id = extract_arxiv_id(arxiv_input)
        if not arxiv_id:
            st.error("Invalid arXiv ID or URL format")
            return False

        # Processing status container
        status_container = st.empty()
        progress_container = st.empty()
        step_container = st.empty()

        with status_container.container():
            st.info("🔄 Processing arXiv paper...")
            st.session_state.start_time = time.time()

            # Fetch paper information
            with progress_container:
                paper_info = load_from_arxiv(arxiv_id)

            step_container.success("✅ Paper retrieved successfully")

            # Store paper information
            st.session_state.current_paper = arxiv_id
            st.session_state.paper_metadata = {
                "type": "arxiv",
                "id": arxiv_id,
                "title": paper_info["title"],
                "authors": paper_info["authors"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.paper_content = paper_info["full_text"]

            # Final success message
            status_container.success(
                f"""
                ### ✅ arXiv Paper Processing Complete
                - 📑 Title: {paper_info["title"]}
                - 👥 Authors: {', '.join(paper_info["authors"])}
                - 📅 Published: {paper_info["published"]}
                - ⌛ Processing Time: {format_time(time.time() - st.session_state.start_time)}
                
                You can now ask questions about the paper!
            """
            )

            # Add initial analysis message
            welcome_msg = f"""I've successfully retrieved and processed the paper "{paper_info['title']}". 
            You can ask me to:
            - Summarize the entire paper
            - Explain specific sections
            - Translate parts of the paper
            - Clarify technical concepts
            - Compare with other research
            
            What would you like to know about the paper?"""

            st.session_state.messages.append(
                {"role": "assistant", "content": welcome_msg}
            )

        return True

    except Exception as e:
        handle_error(e, "arXiv processing")
        return False


def setup_page():
    """Initialize page layout and styling"""
    st.set_page_config(
        page_title="Research Paper Analysis Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/yourusername/research-assistant",
            "Report a bug": "https://github.com/yourusername/research-assistant/issues",
            "About": "Research Paper Analysis Assistant powered by OpenAI",
        },
    )

    # Custom CSS with improved styling
    st.markdown(
        """
        <style>
        .main { padding: 2rem; }
        .stButton>button { 
            width: 100%; 
            margin: 0.5rem 0;
            background-color: #4CAF50;
            color: white;
        }
        .message-user {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .message-assistant {
            background-color: #f5f5f5;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .paper-info {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border: 1px solid #e0e0e0;
        }
        .file-uploader {
            border: 2px dashed #ccc;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .arxiv-input {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .api-key-container {
            background-color: #e8f5e9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def main():
    """Main application logic with improved UI"""
    try:
        setup_page()
        init_session_state()

        # Sidebar
        with st.sidebar:
            st.title("📚 Research Paper Analysis")
            st.markdown("---")

            # API Key handling
            with st.expander(
                "🔑 API Settings", expanded=not st.session_state.api_key_verified
            ):
                if not st.session_state.api_key_verified:
                    st.markdown(
                        """
                        <div class="api-key-container">
                        <h4>OpenAI API Key Required</h4>
                        <p>Please enter your API key to start analyzing papers.</p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    api_key = st.text_input(
                        "OpenAI API Key",
                        type="password",
                        help="Enter your OpenAI API key",
                        key="api_key_input",
                    )

                    if st.button("Verify API Key", key="verify_key"):
                        with st.spinner("Verifying API key..."):
                            if is_valid_openai_api_key(api_key):
                                st.session_state.openai_client = OpenAI(api_key=api_key)
                                st.session_state.api_key_verified = True
                                st.session_state.assistant_id = create_assistant(
                                    st.session_state.openai_client
                                )
                                st.session_state.thread_id = (
                                    st.session_state.openai_client.beta.threads.create().id
                                )
                                st.success("✅ API key verified!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Invalid API key")
                else:
                    st.success("✅ API key verified")
                    if st.button("Change API Key"):
                        st.session_state.api_key_verified = False
                        st.session_state.openai_client = None
                        st.rerun()

            # Conversation controls
            st.markdown("### 💬 Conversation")
            if st.session_state.messages:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.messages = []
                    st.session_state.current_paper = None
                    st.session_state.paper_content = None
                    st.rerun()

            # Help section
            with st.expander("ℹ️ Help & Tips", expanded=False):
                st.markdown(
                    """
                    ### Using the Assistant
                    1. Upload a research paper (PDF)
                    2. Or provide an arXiv paper ID
                    3. Wait for processing to complete
                    4. Ask questions about the paper
                    
                    ### Features
                    - Detailed paper analysis
                    - Korean translation
                    - Technical concept explanation
                    - Interactive Q&A
                    
                    ### Example Questions
                    - "Can you summarize this paper?"
                    - "Explain the methodology in detail"
                    - "Translate this section to Korean"
                    - "What are the key contributions?"
                """
                )

        # Main content
        if not st.session_state.api_key_verified:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to start.")
            return

        st.title("Research Paper Analysis Assistant")

        # Paper Input Section
        if not st.session_state.current_paper:
            st.markdown(
                """
                ### 📄 Upload Research Paper
                Choose one of the following options:
            """
            )

            input_method = st.radio(
                "Select Input Method",
                ["PDF Upload", "arXiv ID"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if input_method == "PDF Upload":
                uploaded_file = st.file_uploader(
                    "Upload PDF", type=["pdf"], help="Maximum file size: 200MB"
                )
                if uploaded_file:
                    handle_paper_upload(uploaded_file)
                    st.rerun()

            else:  # arXiv ID
                arxiv_col1, arxiv_col2 = st.columns([3, 1])
                with arxiv_col1:
                    arxiv_input = st.text_input("arXiv ID:", help="Example: 2312.00752")
                with arxiv_col2:
                    if arxiv_input and st.button("Load Paper"):
                        handle_arxiv_input(arxiv_input)
                        st.rerun()

        # Display current paper info if available
        if st.session_state.current_paper:
            st.markdown("""---""")
            if st.session_state.paper_metadata["type"] == "pdf":
                st.markdown(
                    f"""
                    <div class="paper-info">
                        <h4>📄 Current Paper</h4>
                        <p>Filename: {st.session_state.paper_metadata['name']}</p>
                        <p>Size: {st.session_state.paper_metadata['size'] / 1024:.1f} KB</p>
                        <p>Loaded: {st.session_state.paper_metadata['timestamp']}</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="paper-info">
                        <h4>📑 Current Paper (arXiv)</h4>
                        <p>Title: {st.session_state.paper_metadata['title']}</p>
                        <p>Authors: {', '.join(st.session_state.paper_metadata['authors'])}</p>
                        <p>ID: {st.session_state.paper_metadata['id']}</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

        # Chat Interface
        st.markdown("### 💬 Discussion")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about the paper or request translation..."):
            if not st.session_state.current_paper:
                st.error("Please upload a PDF or provide an arXiv ID first")
            else:
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get assistant response
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Analyzing..."):
                            response = process_message(
                                st.session_state.openai_client,
                                st.session_state.thread_id,
                                st.session_state.assistant_id,
                                prompt,
                            )
                        st.markdown(response)

                        st.session_state.messages.append(
                            {"role": "user", "content": prompt}
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    except Exception as e:
                        handle_error(e, "Question processing")

        # Help section at the bottom
        with st.expander("ℹ️ Help & Tips", expanded=False):
            st.markdown(
                """
                ### Tips for asking questions:
                - Ask about specific sections (abstract, methodology, results)
                - Request comparisons with other papers
                - Ask for explanations of technical concepts
                - Inquire about limitations and future work
            """
            )

    except Exception as e:
        handle_error(e, "Application error")
        st.error("An unexpected error occurred. Please refresh the page and try again.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        handle_error(e, "Application startup")
        st.error("Failed to start the application. Please check the logs for details.")
