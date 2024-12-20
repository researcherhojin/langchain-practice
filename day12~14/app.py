import streamlit as st
import logging
import sys
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("quiz_app.log")],
)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="QuizGPT - AI Quiz Generator", page_icon="🎯", layout="centered"
)


# OpenAI API 키 검증 함수
def is_valid_openai_api_key(api_key: str) -> bool:
    if not api_key or not isinstance(api_key, str):
        return False
    if not api_key.startswith("sk-") or len(api_key) < 20:
        return False
    return True


# 퀴즈 생성 함수 스키마
quiz_function = {
    "name": "create_quiz",
    "description": "Creates a multiple choice quiz with specified parameters",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "correct_answer": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": [
                        "question",
                        "options",
                        "correct_answer",
                        "explanation",
                    ],
                },
            }
        },
        "required": ["questions"],
    },
}


# 퀴즈 생성 함수
def generate_quiz(
    topic: str, num_questions: int, difficulty: str, api_key: str
) -> List[Dict[str, Any]]:
    try:
        logger.info(
            f"Generating quiz - Topic: {topic}, Questions: {num_questions}, Difficulty: {difficulty}"
        )

        # LLM 설정
        llm = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18", temperature=0.7, openai_api_key=api_key
        ).bind(functions=[quiz_function], function_call={"name": "create_quiz"})

        # 프롬프트 템플릿
        template = """Create a multiple choice quiz about {topic}.
        Difficulty level: {difficulty}
        Number of questions: {num_questions}
        
        Requirements:
        - Each question should have exactly 4 options
        - Include clear explanations for the correct answers
        - Make questions {difficulty} difficulty
        - Ensure questions are engaging and test understanding
        - Each question must have exactly one correct answer
        - All options must be unique
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm

        response = chain.invoke(
            {"topic": topic, "difficulty": difficulty, "num_questions": num_questions}
        )

        result = eval(response.additional_kwargs["function_call"]["arguments"])
        logger.info("Successfully generated quiz")
        return result["questions"]

    except Exception as e:
        logger.error(f"Error generating quiz: {str(e)}", exc_info=True)
        raise


# 점수 계산 함수
def calculate_score(
    quiz_data: List[Dict[str, Any]], user_answers: Dict[int, str]
) -> float:
    if not quiz_data or not user_answers:
        return 0.0

    correct_answers = sum(
        1
        for i, q in enumerate(quiz_data)
        if i in user_answers and user_answers[i] == q["correct_answer"]
    )

    total_questions = len(quiz_data)
    return min(100.0, (correct_answers / total_questions) * 100)


# 초기 세션 상태 설정
def initialize_session_state():
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "quiz_completed" not in st.session_state:
        st.session_state.quiz_completed = False
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if "answer_status" not in st.session_state:
        st.session_state.answer_status = {}


# 다음 문제로 이동하는 함수
def next_question():
    if st.session_state.current_question < len(st.session_state.quiz_data) - 1:
        st.session_state.current_question += 1
        st.session_state.submitted = False


# 퀴즈 초기화 함수
def reset_quiz():
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.user_answers = {}
    st.session_state.quiz_completed = False
    st.session_state.submitted = False
    st.session_state.answer_status = {}


# 사이드바 설정
def setup_sidebar():
    with st.sidebar:
        st.title("🎯 QuizGPT")

        # OpenAI API 키 입력
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Get your API key from https://platform.openai.com/account/api-keys",
        )

        # API 키 상태 표시
        if api_key:
            if is_valid_openai_api_key(api_key):
                st.success("API key provided", icon="✅")
            else:
                st.error("Invalid API key format. It should start with 'sk-'")
                api_key = None
        else:
            st.warning("Please enter your OpenAI API key", icon="⚠️")

        # 깃허브 링크
        st.markdown(
            """
            ### Links
            - [GitHub Repository](https://github.com/researcherhojin/langchain-practice)
            """
        )

        if st.button("Clear Quiz History"):
            reset_quiz()
            st.session_state.quiz_data = None
            st.rerun()

        # 퀴즈 통계 표시 (퀴즈가 진행 중일 때)
        if st.session_state.quiz_data:
            st.markdown("---")
            st.markdown("### Quiz Statistics")
            progress = (st.session_state.current_question + 1) / len(
                st.session_state.quiz_data
            )
            st.progress(progress)
            st.write(
                f"Question: {st.session_state.current_question + 1}/{len(st.session_state.quiz_data)}"
            )
            correct_answers = sum(
                1 for i in st.session_state.answer_status.values() if i
            )
            if st.session_state.answer_status:
                st.write(f"Correct Answers: {correct_answers}")

        return api_key


# 메인 앱
def main():
    initialize_session_state()
    api_key = setup_sidebar()

    st.title("🎯 QuizGPT")
    st.markdown("Generate customized quizzes using AI!")

    # 퀴즈 설정
    topic = st.text_input(
        "Enter the quiz topic:", placeholder="e.g., Python Programming Basics"
    )
    num_questions = st.slider(
        "Number of questions:", min_value=1, max_value=10, value=5
    )
    difficulty = st.selectbox(
        "Select difficulty level:",
        ["Easy", "Medium", "Hard"],
        help="Choose the difficulty level of the quiz",
    )

    # 퀴즈 생성 버튼
    if st.button("Generate Quiz"):
        if not topic:
            st.error("Please enter a topic for the quiz.")
            return
        if not api_key:
            st.error("Please provide a valid OpenAI API key.")
            return

        try:
            with st.spinner("Generating quiz..."):
                st.session_state.quiz_data = generate_quiz(
                    topic, num_questions, difficulty, api_key
                )
                reset_quiz()
                st.rerun()
        except Exception as e:
            st.error(f"Failed to generate quiz: {str(e)}")
            logger.error(f"Quiz generation error: {str(e)}", exc_info=True)
            return

    # 퀴즈 표시
    if st.session_state.quiz_data:
        question_data = st.session_state.quiz_data[st.session_state.current_question]

        # 문제 표시
        st.subheader(question_data["question"])

        # 답변 선택
        answer = st.radio(
            "Choose your answer:",
            question_data["options"],
            key=f"q_{st.session_state.current_question}",
            disabled=st.session_state.submitted,
        )

        # 제출 버튼
        if st.button("Submit Answer", disabled=st.session_state.submitted):
            st.session_state.submitted = True
            st.session_state.user_answers[st.session_state.current_question] = answer

            # 정답 체크
            is_correct = answer == question_data["correct_answer"]
            st.session_state.answer_status[st.session_state.current_question] = (
                is_correct
            )

            if is_correct:
                st.success("Correct! 🎉")
                st.session_state.score += 1
            else:
                st.error(
                    f"Incorrect! The correct answer was: {question_data['correct_answer']}"
                )

            st.info(f"Explanation: {question_data['explanation']}")

            # 다음 문제 또는 결과 표시
            if st.session_state.current_question < len(st.session_state.quiz_data) - 1:
                st.button("Next Question", on_click=next_question)
            else:
                st.session_state.quiz_completed = True

        # 결과 표시
        if st.session_state.quiz_completed and st.session_state.submitted:
            st.markdown("---")
            st.subheader("Quiz Completed!")

            # 최종 점수 계산
            final_score = calculate_score(
                st.session_state.quiz_data, st.session_state.user_answers
            )
            st.write(f"Your final score: {final_score:.1f}%")

            # 정답 분석
            correct_count = sum(
                1 for status in st.session_state.answer_status.values() if status
            )
            st.write(
                f"Correct answers: {correct_count}/{len(st.session_state.quiz_data)}"
            )

            # 만점 시 축하
            if final_score == 100.0:
                st.balloons()
                st.success("Congratulations! Perfect score! 🎉")
            # 재시도 옵션
            else:
                if st.button("Try Again"):
                    reset_quiz()
                    st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error("An unexpected error occurred. Please try again.")
