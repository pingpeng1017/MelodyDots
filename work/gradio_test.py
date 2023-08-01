import numpy as np
import gradio as gr
import oemer.ete as ete
import os
import sys as _sys

from oemer import MODULE_PATH
from oemer.utils import get_logger
import argparse

import music21
from music21 import *    #pip install music21


def oemer(input_img):
    # parser = ete.get_parser() # 명령줄 인수를 파싱하는 파서 객체 생성
    # args = parser.parse_args() # 명령줄 인수를 파싱하여 가져옴
    args = argparse.Namespace(img_path=input_img, output_path='./', use_tf=False, save_cache=False, without_deskew=False)

    print("args.img_path:" + args.img_path)

    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"The given image path doesn't exists: {args.img_path}") # 지정된 이미지 경로가 존재하지 않으면 예외 발생

    # Check there are checkpoints
    chk_path = os.path.join(MODULE_PATH, "checkpoints/unet_big/model.onnx")
    if not os.path.exists(chk_path):
        ete.logger.warn("No checkpoint found in %s", chk_path) # 체크포인트가 없는 경우 경고 메시지 출력
        for idx, (title, url) in enumerate(ete.CHECKPOINTS_URL.items()):
            ete.logger.info(f"Downloading checkpoints ({idx+1}/{len(ete.CHECKPOINTS_URL)})") # 체크포인트 다운로드 중인 경우 로그 메시지 출력 
            save_dir = "unet_big" if title.startswith("1st") else "seg_net" # 체크포인트 저장 디렉토리 결정
            save_dir = os.path.join(MODULE_PATH, "checkpoints", save_dir)
            save_path = os.path.join(save_dir, title.split("_")[1]) # 체크포인트 저장 경로 결정
            ete.download_file(title, url, save_path) # 체크포인트 파일 다운로드 함수 호출

    ete.clear_data() # 이전에 등록된 레이어 데이터 모두 삭제
    mxl_path = ete.extract(args) # 이미지에서 악보 추출하여 MusicXML 생성 및 반환
    print("mxl_path:" + mxl_path)
    # img = ete.teaser() # 악보 이미지에 바운딩 박스 등 시각화 정보 추가
    # img.save(mxl_path.replace(".musicxml", "_teaser.png")) # 시각화한 이미지 저장

    # mxl_path = "./image.musicxml"

    # # 점자 변환
    # from music21 import braille
    # dataStr = braille.translate.objectToBraille(mxl_path)

    
    brf_path = mxl_path.replace(".musicxml", ".brf")
    print("brf_path:" + brf_path)
    s = music21.converter.parse( mxl_path ).write( "braille", fp=brf_path )

    dataStr = ""
    with open(brf_path, 'rt', encoding='utf-8') as f:
        dataStr = f.read()


    # dataStr = s.read_text

    return dataStr


demo = gr.Interface(oemer, gr.Image(type="filepath",shape=(300, 600)), "text").queue()
demo.launch()