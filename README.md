# MelodyDots🎶

### 프로젝트 개요

- 프로젝트 기간: 2023.06.30. ~ 2023.08.10.
- 프로젝트 이름 및 제목: MelodyDots - 시각장애인을 위한 악보 점자 변환 시스템
- 프로젝트 설명 및 소개:

  음악 점자 변환 프로그램은 시각 장애인을 위한 음악 접근성을 개선하기 위한 프로젝트입니다.
  <br>이 프로그램은 악보 이미지를 입력으로 받아서 음악 점자로 변환해주는 기능을 제공합니다.

  <p align="center">
    <img src="https://github.com/pingpeng1017/MelodyDots/assets/97069558/b2530d76-819f-4da2-92d2-b5c217910943">
  </p>
  
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
# 프로젝트 디렉토리로 이동한 후 다음 명령어를 실행하여 필요한 종속성과 패키지를 설치한다.
pip install .

# docs/images 폴더에 이미지 파일을 넣어주고, 다음 명령어를 실행한다.
oemer --use-tf --without-deskew <이미지 경로>
```

https://github.com/pingpeng1017/MelodyDots/issues/1#issue-1844313410
