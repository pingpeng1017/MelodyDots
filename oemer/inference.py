import os
import pickle
from PIL import Image

import cv2
import numpy as np

from oemer import MODULE_PATH


# 주어진 이미지의 크기를 조절하여 반환하는 함수
def resize_image(image: Image):
    # Estimate target size with number of pixels.
    # Best number would be 3M~4.35M pixels.
    # 특정 픽셀 수 범위에 해당하지 않는 경우에만 크기를 조정
    w, h = image.size
    pis = w * h
    if 3000000 <= pis <= 435000:
        return image
    lb = 3000000 / pis
    ub = 4350000 / pis
    ratio = pow((lb + ub) / 2, 0.5)
    tar_w = round(ratio * w)
    tar_h = round(ratio * h)
    print(tar_w, tar_h)
    return image.resize((tar_w, tar_h))


# 모델과 이미지를 이용하여 추론을 수행하는 함수
def inference(model_path, img_path, step_size=128, batch_size=16, manual_th=None, use_tf=False):
    if use_tf:
        import tensorflow as tf

        # Tensorflow를 사용하여 모델을 로드하고 가중치를 설정
        arch_path = os.path.join(model_path, "arch.json") # 모델 아키텍처 파일 경로 생성
        w_path = os.path.join(model_path, "weights.h5") # 모델 가중치 파일 경로 생성
        model = tf.keras.models.model_from_json(open(arch_path, "r").read()) # 모델 아키텍처 로드
        model.load_weights(w_path) # 모델 가중치 로드
        input_shape = model.input_shape
        output_shape = model.output_shape 
    else:
        import onnxruntime as rt

        # Onnxruntime을 사용할 경우
        onnx_path = os.path.join(model_path, "model.onnx") # 추론에 사용할 ONNX 모델 파일 경로 생성
        metadata = pickle.load(open(os.path.join(model_path, "metadata.pkl"), "rb")) # 모델의 메타데이터 로드
        providers = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] # 실행 프로바이더들을 지정 (실행 환경 정의)
        sess = rt.InferenceSession(onnx_path, providers=providers) # ONNX 모델을 로드하고 추론을 수행할 세션 생성
        output_names = metadata['output_names']
        input_shape = metadata['input_shape'] 
        output_shape = metadata['output_shape']

    # Collect data
    # 이미지 데이터 수집
    # Tricky workaround to avoid random mistery transpose when loading with 'Image'.
    # 'Image'로 로드할 때 무작위로 발생하는 전치(transpose) 오류를 피하기 위한 꼼수
    image = cv2.imread(img_path)
    image = Image.fromarray(image).convert("RGB")
    image = np.array(resize_image(image))
    win_size = input_shape[1] # 모델의 입력 형태에서 윈도우 크기를 가져옴
    data = [] # 추론에 사용할 데이터를 담을 리스트
    for y in range(0, image.shape[0], step_size): # 이미지의 세로 방향으로 스텝 사이즈로 이동
        if y + win_size > image.shape[0]:
            y = image.shape[0] - win_size
        for x in range(0, image.shape[1], step_size): # 이미지의 가로 방향으로 스텝 사이즈로 이동
            if x + win_size > image.shape[1]:
                x = image.shape[1] - win_size
            hop = image[y:y+win_size, x:x+win_size] # 윈도우 내의 데이터 추출하여 'hop'에 저장
            data.append(hop)

    # Predict
    pred = [] # 추론 결과를 담을 리스트
    for idx in range(0, len(data), batch_size): # 데이터를 배치 크기 단위로 나누어 추론 수행
        print(f"{idx+1}/{len(data)} (step: {batch_size})", end="\r") # 진행 상황 출력
        batch = np.array(data[idx:idx+batch_size]) # 배치 데이터 추출
        # 모델에 배치 데이터를 입력하여 추론 수행 (TensorFlow 모델일 경우 'model.predict()', ONNXRuntime 모델일 경우 'sess.run()')
        out = model.predict(batch) if use_tf else sess.run(output_names, {'input': batch})[0]
        pred.append(out)

    # Merge prediction patches
    output_shape = image.shape[:2] + (output_shape[-1],) # 이미지의 형태와 출력 형태의 마지막 차원을 합침
    out = np.zeros(output_shape, dtype=np.float32) # 출력을 담을 배열을 초기화
    mask = np.zeros(output_shape, dtype=np.float32) # 마스크를 담을 배열을 초기화
    hop_idx = 0 # 추론 결과의 인덱스 초기화
    for y in range(0, image.shape[0], step_size):
        if y + win_size > image.shape[0]:
            y = image.shape[0] - win_size
        for x in range(0, image.shape[1], step_size):
            if x + win_size > image.shape[1]:
                x = image.shape[1] - win_size
            batch_idx = hop_idx // batch_size
            remainder = hop_idx % batch_size
            hop = pred[batch_idx][remainder] # 해당 위치의 추론 결과 가져오기
            out[y:y+win_size, x:x+win_size] += hop 
            mask[y:y+win_size, x:x+win_size] += 1
            hop_idx += 1

    out /= mask # 출력 배열을 마스크로 나누어 평균 계산
    if manual_th is None: # 수동 임계값이 지정되지 않은 경우
        class_map = np.argmax(out, axis=-1) # 출력 배열에서 가장 높은 값을 가진 클래스 선택
    else:
        assert len(manual_th) == output_shape[-1]-1, f"{manual_th}, {output_shape[-1]}" # 'manual_th'의 길이가 출력 형태의 마지막 차원 크기보다 1 작아야 함을 검사
        class_map = np.zeros(out.shape[:2] + (len(manual_th),))
        for idx, th in enumerate(manual_th): # 수동 임계값을 기준으로 클래스 맵 생성
            class_map[..., idx] = np.where(out[..., idx+1]>th, 1, 0)

    return class_map, out # 각 픽셀의 클래스 레이블을 담은 배열과 각 픽셀이 각 클래스에 속할 확률 값


# 주어진 이미지 영역과 모델 이름을 이용하여 예측을 수행하는 함수
def predict(region, model_name):
    if np.max(region) == 1: # 입력 이미지 픽셀 값 범위가 [0, 1]로 정규화된 값인지 확인
        region *= 255 # 정규화된 값인 경우, [0, 255] 범위로 변환
    m_info = pickle.load(open(os.path.join(MODULE_PATH, f"sklearn_models/{model_name}.model"), "rb")) # 저장된 모델 정보 로드
    model = m_info['model']
    w = m_info['w'] # 이미지의 가로 크기
    h = m_info['h'] # 이미지의 세로 크기
    region = Image.fromarray(region.astype(np.uint8)).resize((w, h)) # 입력 이미지를 PIL Image로 변환하고, 모델의 입력 크기에 맞게 리사이징
    pred = model.predict(np.array(region).reshape(1, -1)) # 입력 이미지를 1차원 배열로 펼친 후, 모델로 예측 수행
    return m_info['class_map'][pred[0]] # 예측 결과에 해당하는 클래스 맵을 참조하여 예측 결과 반환


if __name__ == "__main__":
    img_path = "/home/kohara/omr/test_imgs/wind2.jpg"
    model_path = "./checkpoints/seg_net"
    class_map, out = inference(model_path, img_path)