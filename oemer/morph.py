import numpy as np
import cv2


# 커널을 생성하는 함수
def get_kernel(kernel):
    if isinstance(kernel, tuple):
        # 만약 `kernel`이 튜플인 경우, 커널의 모양을 나타냄
        kernel = np.ones(kernel, dtype=np.uint8)
    return kernel


# 작은 구조 요소를 제거하는 연산
def morph_open(img, kernel):
    ker = get_kernel(kernel)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, ker)


# 작은 구조 요소를 채우는 연산
def morph_close(img, kernel):
    ker = get_kernel(kernel)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_CLOSE, ker)


# 특정 패턴을 찾거나 객체의 특정 조건을 충족하는 영역 추출
def morph_hit_miss(img, kernel):
    ker = get_kernel(kernel)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_HITMISS, ker)