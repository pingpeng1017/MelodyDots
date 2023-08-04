import numpy as np
import gradio as gr
import os
import sys as _sys

import oemer.ete as ete
from oemer import MODULE_PATH
from oemer.utils import get_logger
import argparse

import music21
from music21 import *    #pip install music21


def oemer( input_img, input_name ) :
    print( "input_img  : " + input_img )
    print( "input_name : " + input_name )

    # 입력받은 image의 file명을 입력받은 text로 변경
    input_name_path = input_img.replace( "image", input_name )
    print( "input_name_path : " + input_name_path )

    from shutil import copyfile
    # 입력받은 text명으로 복사 생성
    copyfile( input_img, input_name_path )

    # ete.extract()를 호출하기 위한 argument 생성
    args = argparse.Namespace( img_path=input_name_path, output_path='./', use_tf=False, save_cache=False, without_deskew=False )

    print( "args.img_path : " + args.img_path )

    # 지정된 이미지 경로가 존재하지 않으면 예외 발생
    if not os.path.exists( args.img_path ) :
        raise FileNotFoundError( f"The given image path doesn't exists: {args.img_path}" )

    # Check there are checkpoints
    chk_path = os.path.join( MODULE_PATH, "checkpoints/unet_big/model.onnx" )
    if not os.path.exists( chk_path ) :
        # 체크포인트가 없는 경우 경고 메시지 출력
        ete.logger.warn( "No checkpoint found in %s", chk_path )
        for idx, ( title, url ) in enumerate( ete.CHECKPOINTS_URL.items() ) :
            # 체크포인트 다운로드 중인 경우 로그 메시지 출력 
            ete.logger.info( f"Downloading checkpoints ( {idx + 1} / {len( ete.CHECKPOINTS_URL )})" ) 
            # 체크포인트 저장 디렉토리 결정
            save_dir = "unet_big" if title.startswith( "1st" ) else "seg_net" 
            save_dir = os.path.join( MODULE_PATH, "checkpoints", save_dir )
            # 체크포인트 저장 경로 결정
            save_path = os.path.join( save_dir, title.split( "_" )[1] )
            # 체크포인트 파일 다운로드 함수 호출
            ete.download_file( title, url, save_path ) 

    # 이전에 등록된 레이어 데이터 모두 삭제
    ete.clear_data() 
    # 이미지에서 악보 추출하여 MusicXML 생성 및 반환
    mxl_path = ete.extract( args ) 
    print( "mxl_path : " + mxl_path )
    # 악보 이미지에 바운딩 박스 등 시각화 정보 추가
    img = ete.teaser() 
    # 시각화한 이미지 저장
    img.save( mxl_path.replace( ".musicxml", "_teaser.png" ) ) 

    # 점자 변환 : 화면에 보여주기 위한
    txt_path = mxl_path.replace( ".musicxml", ".txt" )
    print( "txt_path : " + txt_path )
    t = music21.converter.parse( mxl_path ).write( "braille", fp=txt_path )

    # 점자 변환 : 파일 다운로드용
    brf_path = mxl_path.replace( ".musicxml", ".brf" )
    print( "brf_path : " + brf_path )
    s = music21.converter.parse( mxl_path ).write( "braille.ascii", fp=brf_path )
 
    # 파일을 읽어 온다
    view_brf = ""
    with open( txt_path, 'rt', encoding='utf-8' ) as f :
        view_brf = f.read()

    return view_brf, brf_path


demo = gr.Interface(
    fn=oemer,
    inputs=[gr.Image( type="filepath" ), "text"],
    outputs=["text", "file"],
).queue() # timeout error 방지를 위해 
demo.launch()