import os
import pickle
import argparse
import urllib.request
from pathlib import Path
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

from oemer import MODULE_PATH
from oemer import layers
from oemer.inference import inference
from oemer.utils import get_logger
from oemer.dewarp import estimate_coords, dewarp
from oemer.staffline_extraction import extract as staff_extract
from oemer.notehead_extraction import extract as note_extract
from oemer.note_group_extraction import extract as group_extract
from oemer.symbol_extraction import extract as symbol_extract
from oemer.rhythm_extraction import extract as rhythm_extract
from oemer.build_system import MusicXMLBuilder
from oemer.draw_teaser import teaser


logger = get_logger(__name__)


# 체크포인트 파일의 존재 여부 확인을 위한 딕셔너리
CHECKPOINTS_URL = {
    "1st_model.onnx": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/1st_model.onnx",
    "1st_weights.h5": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/1st_weights.h5",
    "2nd_model.onnx": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/2nd_model.onnx",
    "2nd_weights.h5": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/2nd_weights.h5"
}


# 이전에 등록된 레이어 데이터를 모두 삭제하는 함수
def clear_data():
    lls = layers.list_layers()
    for l in lls:
        layers.delete_layer(l)


# 주어진 이미지에 대해 악보의 줄과 기호 정보를 추출하는 함수
def generate_pred(img_path, use_tf=False):
    logger.info("Extracting staffline and symbols")
    # 악보의 줄과 기호 추출 작업 시작
    staff_symbols_map, _ = inference(
        os.path.join(MODULE_PATH, "checkpoints/unet_big"), # unet_big 모델 경로
        img_path, # 이미지 경로
        use_tf=use_tf,
    )
    staff = np.where(staff_symbols_map==1, 1, 0) # 값이 1인 픽셀은 악보의 줄로 표시
    symbols = np.where(staff_symbols_map==2, 1, 0) # 값이 2인 픽셀은 기호로 표시

    logger.info("Extracting layers of different symbols")
    # 다른 기호들의 레이어 추출 작업 시작
    symbol_thresholds = [0.5, 0.4, 0.4] # 기호를 추출하기 위한 임계값의 리스트
    sep, _ = inference(
        os.path.join(MODULE_PATH, "checkpoints/seg_net"), # seg_net 모델 경로
        img_path, # 이미지 경로
        manual_th=None,
        use_tf=use_tf,
    )
    stems_rests = np.where(sep==1, 1, 0) # 값이 1인 픽셀은 쉼표로 표시
    notehead = np.where(sep==2, 1, 0) # 값이 2인 픽셀은 음표로 표시
    clefs_keys = np.where(sep==3, 1, 0) # 값이 3인 픽셀은 조표로 표시
    # stems_rests = sep[..., 0]
    # notehead = sep[..., 1]
    # clefs_keys = sep[..., 2]

    return staff, symbols, stems_rests, notehead, clefs_keys


# 기호 예측 결과를 정제하는 함수
def polish_symbols(rgb_black_th=300):
    img = layers.get_layer('original_image') # 원본 이미지 가져오기
    sym_pred = layers.get_layer('symbols_pred') # 기호 예측 결과 가져오기

    img = Image.fromarray(img).resize((sym_pred.shape[1], sym_pred.shape[0])) # 이미지 크기 조정
    arr = np.sum(np.array(img), axis=-1) # 그레이스케일 이미지로 변환
    arr = np.where(arr < rgb_black_th, 1, 0) # 배경 필터링을 위한 이진화
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)) # staff line 필터링을 위한 구조 요소 생성
    arr = cv2.dilate(cv2.erode(arr.astype(np.uint8), ker), ker) # filter staff line 
    mix = np.where(sym_pred+arr>1, 1, 0) # 기호 예측 결과와 배경 필터링 결과를 합성하여 최종 기호 영역 생성
    return mix


# 기호가 있는 영역에 바운딩 박스를 추가해주는 함수
def register_notehead_bbox(bboxes):
    symbols = layers.get_layer('symbols_pred') # 기호 예측 결과 가져오기
    layer = layers.get_layer('bboxes') # bboxes 레이어 가져오기
    for (x1, y1, x2, y2) in bboxes:
        yi, xi = np.where(symbols[y1:y2, x1:x2]>0) # 기호가 있는 픽셀 좌표 가져오기
        yi += y1 # 전체 이미지 좌표로 변환
        xi += x1
        layer[yi, xi] = np.array([x1, y1, x2, y2]) # 기호가 있는 영역에 바운딩 박스 정보 등록
    return layer


# 기호에 대한 노트 ID를 등록하는 함수
def register_note_id():
    symbols = layers.get_layer('symbols_pred') # 기호 예측 결과 가져오기
    layer = layers.get_layer('note_id') # note_id 레이어 가져오기
    notes = layers.get_layer('notes') # notes 레이어 가져오기
    for idx, note in enumerate(notes):
        x1, y1, x2, y2 = note.bbox # 기호의 바운딩 박스 좌표 가져오기
        yi, xi = np.where(symbols[y1:y2, x1:x2]>0) # 기호가 있는 픽셀 좌표 가져오기
        yi += y1 # 전체 이미지 좌표로 변환
        xi += x1
        layer[yi, xi] = idx # 기호가 있는 영역에 해당하는 픽셀에 노트 ID 등록
        notes[idx].id = idx # 노트 객체에 ID 등록


# 이미지로부터 악보 정보를 추출하고 MusicXML을 생성하는 함수
def extract(args):
    img_path = Path(args.img_path)
    f_name = os.path.splitext(img_path.name)[0]
    pkl_path = img_path.parent / f"{f_name}.pkl"
    if pkl_path.exists():
        # Load from cache
        pred = pickle.load(open(pkl_path, "rb"))
        notehead = pred["note"]
        symbols = pred["symbols"]
        staff = pred["staff"]
        clefs_keys = pred["clefs_keys"]
        stems_rests = pred["stems_rests"]
    else:
        # Make predictions
        if args.use_tf:
            ori_inf_type = os.environ.get("INFERENCE_WITH_TF", None)
            os.environ["INFERENCE_WITH_TF"] = "true"
        staff, symbols, stems_rests, notehead, clefs_keys = generate_pred(str(img_path), use_tf=args.use_tf)
        if args.use_tf and ori_inf_type is not None:
            os.environ["INFERENCE_WITH_TF"] = ori_inf_type
        if args.save_cache:
            data = {
                'staff': staff,
                'note': notehead,
                'symbols': symbols,
                'stems_rests': stems_rests,
                'clefs_keys': clefs_keys
            }
            pickle.dump(data, open(pkl_path, "wb"))

    # Load the original image, resize to the same size as prediction
    image = cv2.imread(str(img_path))
    image = cv2.resize(image, (staff.shape[1], staff.shape[0]))

    if not args.without_deskew:
        logger.info("Dewarping")
        # 변형 보정 작업 시작
        coords_x, coords_y = estimate_coords(staff)
        staff = dewarp(staff, coords_x, coords_y)
        symbols = dewarp(symbols, coords_x, coords_y) 
        stems_rests = dewarp(stems_rests, coords_x, coords_y)
        clefs_keys = dewarp(clefs_keys, coords_x, coords_y)
        notehead = dewarp(notehead, coords_x, coords_y)
        for i in range(image.shape[2]):
            image[..., i] = dewarp(image[..., i], coords_x, coords_y) # 원본 이미지 보정

    # Register predictions
    symbols = symbols + clefs_keys + stems_rests
    symbols[symbols>1] = 1
    layers.register_layer("stems_rests_pred", stems_rests)
    layers.register_layer("clefs_keys_pred", clefs_keys)
    layers.register_layer("notehead_pred", notehead)
    layers.register_layer("symbols_pred", symbols)
    layers.register_layer("staff_pred", staff)
    layers.register_layer("original_image", image)

    # ---- Extract staff lines and group informations ---- #
    logger.info("Extracting stafflines")
    # 악보 줄 추출 작업 시작
    staffs, zones = staff_extract()
    layers.register_layer("staffs", staffs)  # Array of 'Staff' instances
    layers.register_layer("zones", zones)  # Range of each zones, array of 'range' object

    # ---- Extract noteheads ---- #
    logger.info("Extracting noteheads")
    # 음표 추출 작업 시작
    notes = note_extract()

    # Array of 'NoteHead' instances
    layers.register_layer('notes', np.array(notes))

    # Add a new layer (w * h), indicating note id of each pixel
    layers.register_layer('note_id', np.zeros(symbols.shape, dtype=np.int)-1)
    register_note_id()

    # ---- Extract groups of note ---- #
    logger.info("Grouping noteheads")
    # 음표 그룹화 작업 시작
    groups, group_map = group_extract()
    layers.register_layer('note_groups', np.array(groups))
    layers.register_layer('group_map', group_map)

    # ---- Extract symbols ---- #
    logger.info("Extracting symbols")
    # 기호 추출 작업 시작
    barlines, clefs, sfns, rests = symbol_extract()
    layers.register_layer('barlines', np.array(barlines))
    layers.register_layer('clefs', np.array(clefs))
    layers.register_layer('sfns', np.array(sfns))
    layers.register_layer('rests', np.array(rests))

    # ---- Parse rhythm ---- #
    logger.info("Extracting rhythm types")
    # 리듬 유형 추출 작업 시작
    rhythm_extract()

    # ---- Build MusicXML ---- #
    logger.info("Building MusicXML document")
    # MusicXML 문서 생성 작업 시작
    basename = os.path.basename(img_path).replace(".jpg", "").replace(".png", "")
    builder = MusicXMLBuilder(title=basename.capitalize())
    builder.build()
    xml = builder.to_musicxml()

    # ---- Write out the MusicXML ---- #
    out_path = args.output_path
    if not out_path.endswith(".musicxml"):
        # Take the output path as the folder
        out_path = os.path.join(out_path, basename+".musicxml")

    with open(out_path, "wb") as ff:
        ff.write(xml)

    return out_path


# 커맨드 라인 도구에서 사용할 수 있는 인자 파서를 생성하는 함수
def get_parser():
    parser = argparse.ArgumentParser(
        "Oemer",
        description="End-to-end OMR command line tool. Receives an image as input, and outputs MusicXML file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("img_path", help="Path to the image.", type=str)
    parser.add_argument(
        "-o", "--output-path", help="Path to output the result file.", type=str, default="./")
    parser.add_argument(
        "--use-tf", help="Use Tensorflow for model inference. Default is to use Onnxruntime.", action="store_true")
    parser.add_argument(
        "--save-cache",
        help="Save the model predictions and the next time won't need to predict again.",
        action='store_true')
    parser.add_argument(
        "-d",
        "--without-deskew",
        help="Disable the deskewing step if you are sure the image has no skew.",
        action='store_true')
    return parser


# 주어진 URL로부터 파일을 다운로드하는 함수
def download_file(title, url, save_path):
    resp = urllib.request.urlopen(url)
    length = int(resp.getheader("Content-Length", -1))

    chunk_size = 2**9
    total = 0
    with open(save_path, "wb") as out:
        while True:
            print(f"{title}: {total*100/length:.1f}% {total}/{length}", end="\r")
            data = resp.read(chunk_size)
            if not data:
                break
            total += out.write(data)
        print(f"{title}: 100% {length}/{length}"+" "*20)


def main():
    parser = get_parser() # 명령줄 인수를 파싱하는 파서 객체 생성
    args = parser.parse_args() # 명령줄 인수를 파싱하여 가져옴

    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"The given image path doesn't exists: {args.img_path}") # 지정된 이미지 경로가 존재하지 않으면 예외 발생

    # Check there are checkpoints
    chk_path = os.path.join(MODULE_PATH, "checkpoints/unet_big/model.onnx")
    if not os.path.exists(chk_path):
        logger.warn("No checkpoint found in %s", chk_path) # 체크포인트가 없는 경우 경고 메시지 출력
        for idx, (title, url) in enumerate(CHECKPOINTS_URL.items()):
            logger.info(f"Downloading checkpoints ({idx+1}/{len(CHECKPOINTS_URL)})") # 체크포인트 다운로드 중인 경우 로그 메시지 출력 
            save_dir = "unet_big" if title.startswith("1st") else "seg_net" # 체크포인트 저장 디렉토리 결정
            save_dir = os.path.join(MODULE_PATH, "checkpoints", save_dir)
            save_path = os.path.join(save_dir, title.split("_")[1]) # 체크포인트 저장 경로 결정
            download_file(title, url, save_path) # 체크포인트 파일 다운로드 함수 호출

    clear_data() # 이전에 등록된 레이어 데이터 모두 삭제
    mxl_path = extract(args) # 이미지에서 악보 추출하여 MusicXML 생성 및 반환
    img = teaser() # 악보 이미지에 바운딩 박스 등 시각화 정보 추가
    img.save(mxl_path.replace(".musicxml", "_teaser.png")) # 시각화한 이미지 저장


if __name__ == "__main__":
    main()
