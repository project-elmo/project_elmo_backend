<h1 align="center">
  <br>
  <img src="https://github.com/project-elmo/frontend/assets/72433681/2f99edb0-c873-478a-9add-d21eef560ccc" alt="ELMO" width="300">
</h1>

<h4 align="center">
Easy LLM Online Model Optimizer
</h4>
<p align="center">
  <a href="https://github.com/project-elmo/frontend/blob/main/README.md" target="_blank">
    한국어
  </a>
	|
  <a href="https://github.com/project-elmo/frontend/blob/main/README-en.md" target="_blank">English</a>
</p>
<h4 align="center">
  <a href="https://elmo-beta.vercel.app/" target="_blank">
    🌐 배포
  </a> | 
  <a href="https://youtu.be/jVAlqwMY1co?feature=shared" target="_blank">🖥️ 시연</a>
</h4>

<p align="center">
  <a href="#소개">소개</a> •
  <a href="#주요-기능">주요 기능</a> •
  <a href="#프로젝트-구조">프로젝트 구조</a> •
  <a href="#사용-방법">사용 방법</a> •
  <a href="#기술-스택">기술 스택</a> •
  <a href="#참고-자료">참고 자료</a> •
  <a href="#출처-알림">출처 알림</a>
</p>

<div align="center">

| [<img src="https://github.com/hhkim0729.png" width="100px;"/><br /><sub><b>김희현</b></sub>](https://github.com/hhkim0729) | [<img src="https://github.com/jharinn.png" width="100px;"/><br /><sub><b>정하림</b></sub>](https://github.com/jharinn)<br /> |
| :------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: |
|                                                         프론트엔드                                                         |                                                          백엔드, AI                                                          |

</div>

## 소개

인공지능에 대한 배경지식이 없어도 LLM을 학습시켜 나만의 GPT를 만들 수 있는 솔루션 오픈 소스 프로젝트

- 개발 기간: 2023.08 ~ 2023.09 (4주)

### 1. 개발 배경

- **부상하는 생성형 인공지능**

  최근 자연어 처리 기술은 챗지피티와 같은 서비스를 통해 큰 인기를 얻고 있으며, 다양한 분야에서 사용될 것으로 기대된다.

- **거대 언어모델(LLM)의 성장과 그 필요성**

  IT 대기업들이 거대 언어 모델인 LLM을 출시하며 특화된 성능 개선을 위해 파인튜닝의 필요성을 인식하고 있다. 이에 따라 기업 및 개인 맞춤형 LLM의 성장이 예상된다.

- **LLM의 높은 수요와 어려운 접근 환경**

  LLM에 대한 수요는 높아지고 있지만, 대중에게는 접근이 어렵다. 대형 언어 모델은 고가의 GPU가 필요하여 대부분 대기업에서만 개발 가능한 현실이다. 또한 언어 모델을 훈련시키려면 깊은 지식이 요구된다. 이에 따라 간단하게 언어 모델을 파인튜닝할 수 있는 서비스의 필요성이 대두되었다.

- **소규모 언어모델의 부상**

  특정 분야나 목적에 특화된 소규모 언어 모델의 중요성이 강조되고 있다. 이를 통해 하드웨어 비용 없이 특정 분야에 한정된 학습을 가능하게 하고 학습 비용과 시간을 절약할 수 있다. 메타의 연구에 따르면, 파라미터 규모가 적은 모델도 고품질의 학습 데이터와 반복적인 트레이닝으로 큰 모델만큼의 성능을 낼 수 있으며, 인터넷에 공개된 데이터로 훈련하여 개인의 컴퓨터에서도 목적에 맞는 GPT를 만들 수 있다고 강조한다.

### 2. 개발 목적

- **맞춤형 AI 모델의 접근성 향상**

  LLM 학습은 복잡한 과정을 필요로 하며, 일반인에게는 큰 진입 장벽이 될 수 있다. 이러한 장벽을 낮추기 위해 사용자 친화적인 인터페이스와 서비스를 제공하는 것이 본 프로젝트의 주요 목적이다.

- **다양한 분야에서의 높은 활용성**

  본 프로젝트를 통해 개인과 기업 모두가 다양한 분야에서 LLM을 활용할 수 있도록 지원한다. 예를 들어, 교육 분야에서는 맞춤형 교재나 학습 자료 제작에, 마케팅 분야에서는 고객 데이터 분류와 통찰 도출에 LLM을 활용할 수 있다.

- **사용자 중심의 서비스 제공으로 인공지능 기술의 대중화**

  개인과 기업이 특정 목적에 따라 LLM을 쉽게 파인튜닝하고 활용할 수 있는 플랫폼을 제공하여, 인공지능 기술의 일상화와 대중화를 실현하고자 한다.

### 3. 활용 분야

- **마케팅**
  - 고객의 관심사와 요구에 기반한 맞춤형 마케팅 캠페인 개발
  - 블로그 글, 뉴스레터 등 제품에 맞춘 콘텐츠 생성
- **교육**
  - 학생별 맞춤형 학습 자료, 예제 생성
  - 강의, 과제 내용에 대한 질의 응답
- **기업**
  - 기업의 도메인에 특화된 사내 챗봇 서비스
  - 소규모 기업체의 고객지원 챗봇 서비스

## 주요 기능

### 1. 모델 훈련

<img src="https://github.com/project-elmo/frontend/assets/72433681/a5a06711-3546-48ed-a5c2-18a64dc1884f" alt="모델 선택" width="500">

- pre-trained 모델 선택
  - 모델 목록 표시
  - 모델 다운로드

<img src="https://github.com/project-elmo/frontend/assets/72433681/c9eef064-58d9-4d5e-ab44-d1e923cc0c1e" alt="데이터셋 업로드" width="500">

- 데이터셋 업로드
  - JSON, CSV 파일 추가
  - ❗️ 현재 데모에서는 QA(질문-답변) 모델을 훈련할 수 있으며, JSON의 키 값 또는 CSV의 컬럼명이 'question'과 'answer'로 라벨링되어야 함

<img src="https://github.com/project-elmo/frontend/assets/72433681/e52c9c53-f42d-433e-bc8e-ddd5185ca024" alt="파라미터 조정" width="500">
<img src="https://github.com/project-elmo/frontend/assets/72433681/a14531f8-0ab2-40be-9186-dc1d3858ceb9" alt="모델 훈련 과정" width="500">
<img src="https://github.com/project-elmo/frontend/assets/72433681/0f180642-39c2-4dd1-9472-d6d5c68bd4b4" alt="모델 훈련 결과" width="500">

- 모델 훈련
  - LLM 트레이닝 파라미터 조정
  - 모델 훈련 과정 표시
  - 모델 훈련 결과 저장

### 2. 모델 관리

<img src="https://github.com/project-elmo/frontend/assets/72433681/502e230a-0aba-4916-b2ac-7d594f0634b7" alt="이전 훈련 결과" width="500">

- 이전 훈련 결과 조회
  - 모델 관계를 트리 구조로 표시
  - 훈련 모델의 메타데이터 표시
- 이전 모델 훈련 결과 다운로드
  - 가중치 파일 로컬 다운로드
- 이전 모델에 이어서 훈련하기

### 3. 모델 테스트

<img src="https://github.com/project-elmo/frontend/assets/72433681/2897cbb8-ecc7-449b-a69b-2e5a56e6a7ff" alt="테스트할 모델 선택" width="500">
<img src="https://github.com/project-elmo/frontend/assets/72433681/e1bfe440-3b57-4604-ab57-552d8534f29e" alt="채팅 형식 테스트" width="500">

- 텍스트 생성
  - 사용할 모델 선택
  - 채팅 형식 테스트

## 프로젝트 구조

![image](https://github.com/project-elmo/frontend/assets/72433681/aeebd11b-60ed-4872-add0-9aab406ae8a6)

## 사용 방법

[📋 API 문서](https://elmo-demo.store/docs)

### 1. 서버

- Docker 사용:
  ```python
  docker compose build
  docker compose up --build
  ```
- <details close>
    <summary>Docker 없이 사용</summary>
    1. 패키지 추가를 위한 poetry 설치

  ```python
  pip install poetry==1.6.0
  ```

  2. poetry로 필요한 패키지 설치

  ```python
  poetry install
  ```

  3. Redis & MySQL 준비

  ```python
  (macOS)
  brew install redis

  (Linux -e.g., Ubuntu-)
  sudo apt-get install redis-server

  (window)
  you can use WSL or a Redis Windows port.

  redis-server redis.conf

  sudo apt-get update
  sudo apt-get install mysql-server-5.7
  sudo mysql_install_db

  (macOS)
  brew services start mysql@5.7
  # brew services restart mysql@5.7

  (Linux)
  mysql.server start

  mysql -u root -p

  mysql
  **CREATE DATABASE fastapi;
  SET PASSWORD FOR 'root'@'localhost' = PASSWORD('fastapi');**
  ```

  4. alembic으로 DB 설정

  ```python
  alembic upgrade head
  ```

  5. 서버 시작

  ```python
  poetry run python3 main.py --env {env} --debug
  ```

  </details>

### 2. 클라이언트

1. 패키지 추가를 위한 pnpm 설치

   https://pnpm.io/ko/installation

2. 필요한 패키지 설치
   ```sh
   pnpm install
   ```
3. 개발 서버 시작
   ```sh
   pnpm run dev
   ```

## 기술 스택

### 프론트엔드

![react](https://img.shields.io/badge/react-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Tanstack Query](https://img.shields.io/badge/-Tanstack%20Query-FF4154?style=for-the-badge&logo=react%20query&logoColor=white)
![Axios](https://img.shields.io/badge/axios-5A29E4?style=for-the-badge&logo=axios&logoColor=white)
![radixui](https://img.shields.io/badge/radixui-161618?style=for-the-badge&logo=radixui&logoColor=white)
![d3.js](https://img.shields.io/badge/d3.js-F9A03C?style=for-the-badge&logo=d3dotjs&logoColor=white)
![React Flow](https://img.shields.io/badge/React%20Flow-db0070?style=for-the-badge)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)
![React Icons](https://img.shields.io/badge/React%20Icons-c82361?style=for-the-badge)

![typescript](https://img.shields.io/badge/typescript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![vite](https://img.shields.io/badge/vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![ESLint](https://img.shields.io/badge/ESLint-4B3263?style=for-the-badge&logo=eslint&logoColor=white)
![Prettier](https://img.shields.io/badge/Prettier-F7B93E?style=for-the-badge&logo=prettier&logoColor=black)
![pnpm](https://img.shields.io/badge/pnpm-F69220?style=for-the-badge&logo=pnpm&logoColor=white)

### 백엔드

![python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![hugging face](https://img.shields.io/badge/hugging%20face-f3d13b?style=for-the-badge)
![uvicorn](https://img.shields.io/badge/uvicorn-4752b1?style=for-the-badge)
![gunicorn](https://img.shields.io/badge/gunicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white)
![mysql](https://img.shields.io/badge/mysql-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![redis](https://img.shields.io/badge/redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![fastapi](https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![alchemy](https://img.shields.io/badge/alchemy-0C0C0E?style=for-the-badge&logo=alchemy&logoColor=white)
![Alembic](https://img.shields.io/badge/Alembic-black?style=for-the-badge)

![docker](https://img.shields.io/badge/docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

### 개발

![Git](https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

### 협업

![GitHub](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-FFFFFF.svg?style=for-the-badge&logo=notion&logoColor=black)
![Figma](https://img.shields.io/badge/figma-%23F24E1E.svg?style=for-the-badge&logo=figma&logoColor=white)
![swagger](https://img.shields.io/badge/swagger-85EA2D?style=for-the-badge&logo=swagger&logoColor=black)

## 참고 자료

- [[커버스토리] 챗GPT 타고 확산하는 ‘거대언어모델’…sLLM 구축 확대일로](http://www.itdaily.kr/news/articleView.html?idxno=215587)
- [작지만 똑똑한 AI … sLLM 시대 온다](https://www.mk.co.kr/news/it/10791394)
- Touvron, Hugo, et al. "LLaMA: Open and Efficient Foundation Language Models." (2023).

## 출처 알림

- 로고 이미지: [Emoji Kitchen](https://emojikitchen.dev/)
- 백엔드 템플릿: [FastAPI Boilerplate](https://github.com/teamhide/fastapi-boilerplate)
- 시연 영상에 사용된 데모데이터 셋: [CareCall for Seniors](https://github.com/naver-ai/carecall-corpus)
  - 모든 데이터에 대한 모든 권리(저작권 등 지식재산권 포함)는 네이버에게 있습니다
