# LangChain Practice

LangChain을 활용한 21일간의 실습 프로젝트입니다.

## 📝 프로젝트 소개

-   LangChain의 다양한 기능을 실험하고 학습하는 프로젝트
-   매일 다른 주제로 실습을 진행하며 결과물을 기록
-   2024.12.09 - 2024.12.29 (21일간 진행)

## 📅 진행 상황

### Day 01 (2024.12.09) - 프로그래밍 언어 시 생성기

-   ChatPromptTemplate을 활용한 프롬프트 엔지니어링
-   LCEL(LangChain Expression Language)을 활용한 체인 구성
-   주요 프로그래밍 언어(Python, Java, JavaScript 등)에 대한 시 생성
-   생성된 시에 대한 문학적 해석 추가
-   결과물 자동 저장 기능 구현

#### 사용된 주요 기능

-   `ChatPromptTemplate`: 구조화된 프롬프트 생성
-   `ChatOpenAI`: GPT 모델 인터페이스
-   `RunnablePassthrough`: 체인 데이터 흐름 제어
-   `StreamingStdOutCallbackHandler`: 실시간 출력 스트리밍

#### 결과물

-   프로그래밍 언어별 특성을 반영한 시 생성
-   생성된 시에 대한 전문적인 해석
-   자동 저장 및 관리 시스템

## 🛠 기술 스택

-   Python
-   LangChain
-   OpenAI GPT-4
-   Jupyter Notebook

## 📁 프로젝트 구조

```
langchain-practice/
├── README.md
├── .gitignore
├── day01/
│   └── [생성된 시 파일들]
└── day01.ipynb
```

## 🚀 실행 방법

1. 환경 설정

```bash
# 필요한 패키지 설치
pip install langchain langchain-openai
```

2. 환경 변수 설정

```
# .env 파일에 API 키 설정
OPENAI_API_KEY=your_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

3. 노트북 실행

-   Jupyter Notebook에서 day01.ipynb 실행

## 👥 기여 방법

-   Issue 생성: 버그 리포트, 기능 제안
-   Pull Request: 코드 개선, 새로운 기능 추가

## 📌 참고 자료

-   [LangChain 공식 문서](https://python.langchain.com/docs/get_started/introduction)
