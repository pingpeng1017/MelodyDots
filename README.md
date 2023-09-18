# MelodyDots🎶

  <p align="left">
    <img src="https://github.com/pingpeng1017/MelodyDots/assets/97069558/306d898c-a4fb-4ad7-b3ab-e5d3ef83ecb0">
  </p>
  
### 프로젝트 개요

- 프로젝트 기간: 2023.06.30 ~ 2023.08.10
- 프로젝트 이름 및 제목: MelodyDots - 시각 장애인을 위한 음악 악보 자동 점자 변환 시스템
- 프로젝트 설명 및 소개:

  음악 점자 변환 프로그램은 시각 장애인을 위한 음악 접근성을 개선하기 위한 프로젝트입니다.
  <br>이 프로그램은 악보 이미지를 입력으로 받아서 음악 점자로 변환해주는 기능을 제공합니다.
  
### 팀 소개

- 팀원 구성 및 역할:
  - 이민주: 프로젝트 팀장, 모델링 및 학습
  - 정영재: 데이터 수집 및 전처리
  - 한상덕: 알고리즘 및 웹 개발

### 필수 라이브러리 및 프레임워크

- Python 3.8
- numpy
- tensorflow
- scikit-learn
- music21
- gradio

### 설치 및 실행 가이드
```bash
# 프로젝트 디렉토리로 이동
pip install .

# 이미지 파일을 docs/images 폴더에 넣고 실행
oemer --use-tf --without-deskew <이미지 경로>
```
