import os
import pickle

import cv2
import numpy as np
import scipy.ndimage
from scipy.interpolate import interp1d, griddata
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from oemer.morph import morph_open
from oemer.utils import get_logger


# 로깅을 위한 'logger' 객체 생성
logger = get_logger(__name__)


# 격자를 나타내는 객체를 생성하기 위한 클래스
class Grid:
    def __init__(self):
        self.id: int = None # 고유 식별자
        self.bbox: list[int] = None # 경계 상자 (x1, y1, x2, y2)를 나타내는 정수형 리스트
        self.y_shift: int = 0 # y 축 이동

    @property
    # y 중심 좌표를 계산하여 반환
    def y_center(self):
        return (self.bbox[1]+self.bbox[3]) / 2

    @property
    # 높이를 계산하여 반환
    def height(self):
        return self.bbox[3] - self.bbox[1]


# 격자 그룹을 나타내는 객체를 생성하기 위한 클래스
class GridGroup:
    def __init__(self):
        self.id: int = None # 고유 식별자
        self.reg_id: int = None # 레지스터 식별자
        self.bbox: list[int] = None # 경계 상자 (x1, y1, x2, y2)를 나타내는 정수형 리스트
        self.gids: list[int] = [] # 격자의 고유 식별자들을 담은 리스트
        self.split_unit: int = None # 분할 단위

    @property
    # y 중심 좌표를 계산하여 반환 (반올림)
    def y_center(self):
        return round((self.bbox[1]+self.bbox[3]) / 2)
    
    def __lt__(self, tar):
        # Sort by width
        w = self.bbox[2] - self.bbox[0]
        tw = tar.bbox[2] - tar.bbox[0]
        return w < tw

    # 문자열 표현을 반환
    def __repr__(self):
        return f"Grid Group {self.id} / Width: {self.bbox[2]-self.bbox[0]} / BBox: {self.bbox}" \
            f" / Y-center: {self.y_center} / Reg. ID: {self.reg_id}"


# 격자를 생성하는 함수
def build_grid(st_pred, split_unit=11):
    grid_map = np.zeros(st_pred.shape) - 1 # 격자 맵 초기화
    h, w = st_pred.shape # 입력 이미지의 높이와 너비

    is_on = lambda data: np.sum(data) > split_unit//2  # split_unit 기준으로 데이터가 켜져 있는지 확인

    grids = [] # 격자들을 저장할 리스트
    for i in range(0, w, split_unit): # 격자 단위로 열을 이동
        cur_y = 0 # 현재 y 좌표 초기화
        last_y = 0 # 마지막 y 좌표 초기화
        cur_stat = is_on(st_pred[cur_y, i:i+split_unit]) # 현재 열에서의 데이터 상태 확인
        while cur_y < h: # 이미지의 높이까지 반복
            while cur_y < h and cur_stat == is_on(st_pred[cur_y, i:i+split_unit]):
                cur_y += 1 # 동일한 데이터 상태인 동안 y 좌표 이동
            if cur_stat and (cur_y-last_y < split_unit):
                # 데이터가 켜져 있고, split_unit 보다 작은 높이의 경우
                # Switch off
                grid_map[last_y:cur_y, i:i+split_unit] = len(grids) # 격자 맵에 격자 번호 할당
                gg = Grid() # 격자 객체 생성
                gg.bbox = (i, last_y, i+split_unit, cur_y) # 격자의 경계 상자 설정
                gg.id = len(grids) # 격자 번호 설정
                grids.append(gg) # 격자 리스트에 추가
            cur_stat = not cur_stat  # 데이터 상태 변경 (켜짐 <-> 꺼짐)
            last_y = cur_y # 마지막 y 좌표 갱신
    return grid_map, grids # 격자 맵과 격자 리스트 반환


# 격자 그룹을 생성하는 함수
def build_grid_group(grid_map, grids):
    regions, feat_num = scipy.ndimage.label(grid_map+1) # 격자 맵에 레이블링을 수행하여 각 영역을 구분
    grid_groups = []
    for i in range(feat_num):
        # 현재 레이블링된 영역에 해당하는 격자 식별자 추출
        region = grid_map[regions==i+1]
        gids = list(np.unique(region).astype(int))
        gids = sorted(gids)
        
        # 격자 그룹의 경계 상자를 계산
        lbox = grids[gids[0]].bbox # 첫번째 격자의 경계 상자
        rbox = grids[gids[-1]].bbox # 마지막 격자의 경계 상자
        box = (
            min(lbox[0], rbox[0]), # 최소 x 값
            min(lbox[1], rbox[1]), # 최소 y 값
            max(lbox[2], rbox[2]), # 최대 x 값
            max(lbox[3], rbox[3]), # 최대 y 값
        )
        
        # 격자 그룹 객체 생성 및 속성 설정
        gg = GridGroup()
        gg.reg_id = i + 1
        gg.gids = gids
        gg.bbox = box
        gg.split_unit = lbox[2] - lbox[0]
        grid_groups.append(gg)

    # 격자 그룹들을 가로 길이에 따라 내림차순으로 정렬
    grid_groups = sorted(grid_groups, reverse=True)
    gg_map = np.zeros_like(regions) - 1
    for idx, gg in enumerate(grid_groups):
        gg.id = idx
        gg_map[regions==gg.reg_id] = idx
        gg.reg_id = idx

    return gg_map, grid_groups


# 격자 그룹 간의 연결을 수행하는 함수
def connect_nearby_grid_group(gg_map, grid_groups, grid_map, grids, ref_count=8, max_step=20):
    new_gg_map = np.copy(gg_map) # 새로운 격자 그룹 맵을 생성
    ref_gids = grid_groups[0].gids[:ref_count] # 참조 격자 그룹의 격자 식별자들 가져오기
    idx = 0
    gg = grid_groups[idx] # 현재 격자 그룹 초기화
    remaining = set(range(len(grid_groups))) # 방문하지 않은 격자 그룹 식별자들을 추적하기 위한 집합 생성
    # 모든 격자 그룹이 방문될 때까지
    while remaining:
        # Check if not yet visited
        if gg.id not in remaining: 
            # 이미 방문된 격자 그룹인 경우, 다음 처리할 방문하지 않은 격자 그룹 선택
            if remaining:
                gid = remaining.pop()
                gg = grid_groups[gid]
                ref_gids = gg.gids[:ref_count]
            else:
                break
        else:
            # 방문하지 않은 격자 그룹인 경우, 현재 격자 그룹 ID를 방문 완료된 그룹 집합에서 제거
            remaining.remove(gg.id)

        # 참조 격자 그룹에 속한 격자의 개수가 2보다 작은 경우, 현재 격자 그룹 건너뛰기
        if len(ref_gids) < 2:
            continue

        # Extend on the left side
        step_size = gg.split_unit # 현재 격자 그룹의 분할 단위 설정
        centers = [grids[gid].y_center for gid in ref_gids] # 참조 격자 그룹에 속한 격자들의 y 중심 좌표 추출
        x = np.arange(len(centers)).reshape(-1, 1) * step_size # x 좌표 배열 생성
        model = LinearRegression().fit(x, centers) # 선형 회귀 모델 학습
        ref_box = grids[ref_gids[0]].bbox # 참조 격자 그룹의 첫 번째 격자의 경계 상자 추출

        end_x = ref_box[0] # 시작 x 좌표를 참조 격자 그룹의 첫 번째 격자의 x 좌표로 설정
        h = ref_box[3] - ref_box[1] # 참조 격자 그룹의 높이 계산
        cands_box = [] # 잠재적인 경로를 담을 빈 리스트
        for i in range(max_step):
            tar_x = (-i - 1) * step_size # 잠재적인 경로의 x 좌표 계산
            cen_y = model.predict([[tar_x]])[0] # y 중심 좌표를 예측하여 보간
            y = int(round(cen_y - h / 2)) # 경계 상자의 y 좌표 계산
            region = new_gg_map[y:y+h, end_x-step_size:end_x] # Area to check
            unique, counts = np.unique(region, return_counts=True) # 격자 그룹 ID의 고유한 값과 개수 계산
            labels = set(unique) # Overlapped grid group IDs
            if -1 in labels:
                labels.remove(-1) # -1을 겹치는 그룹 ID에서 제거
                
            cands_box.append((end_x-step_size, y, end_x, y+h))  # 잠재적 경로에 상자 추가
            if len(labels) == 0: # 겹치는 그룹이 없는 경우, x 좌표를 이동하여 다음 상자 위치 설정
                end_x -= step_size
            else:
                cands_box = cands_box[:-1] # Remove the overlapped box

                # Determine the overlapped grid group id
                if len(labels) > 1:
                    # 여러 그룹과 겹치는 경우, 겹침 크기를 기준으로 정렬하여 가장 큰 크기를 갖는 그룹 선택
                    overlapped_size = sorted(zip(unique, counts), key=lambda it: it[1], reverse=True)
                    label = overlapped_size[0][0]
                else:
                    # 겹치는 그룹이 한 개인 경우, 해당 그룹 선택
                    label = labels.pop()

                # Check the overlappiong with the traget grid group is valid.
                tar_box = grid_groups[label].bbox
                if tar_box[2] > end_x:
                    break

                # Start assign grid to disconnected position.
                # Get the grid ID.
                yidx, xidx = np.where(region==label)
                yidx += y
                xidx += end_x-step_size
                reg = grid_map[yidx, xidx]
                grid_id, counts = np.unique(reg, return_counts=True)
                if len(grid_id) > 1:
                    logger.warn(
                        "Detected multiple possible overlapping grids: %s. Reg. count: %s",
                        str(grid_id), str(counts))
                grid_id = int(grid_id[np.argmax(counts)])
                assert grid_id in grid_groups[label].gids, f"{grid_id}, {label}"
                grid = grids[grid_id]

                # Interpolate y centers between the start and end points again.
                centers = [grid.y_center, centers[0]]
                x = [-i-1, 0]
                inter_func = interp1d(x, centers, kind='linear')

                # Start to insert grids between points
                cands_ids = []
                for bi, box in enumerate(cands_box):
                    interp_y = round(inter_func(-bi-1) - h/2) # 보간된 y 중심 좌표 계산
                    grid = Grid()
                    box = (box[0], interp_y, box[2], interp_y+h) # 상자 좌표 계산
                    grid.bbox = box
                    grid.id = len(grids)
                    cands_ids.append(len(grids))
                    gg.gids.append(len(grids))
                    gg.bbox = (
                        min(gg.bbox[0], box[0]),
                        min(gg.bbox[1], box[1]),
                        max(gg.bbox[2], box[2]),
                        max(gg.bbox[3], box[3])
                    )
                    # 격자 그룹의 경계 상자 업데이트
                    gg.bbox = [int(bb) for bb in gg.bbox]
                    box = [int(bb) for bb in box]
                    grids.append(grid)
                    new_gg_map[box[1]:box[3], box[0]:box[2]] = gg.id # 격자 그룹 ID로 격자 맵 업데이트

                # Continue to track on the overlapped grid group.
                gg = grid_groups[label]
                gids = gg.gids + cands_ids[::-1]
                ref_gids = gids[:ref_count]

                break

    return new_gg_map


# 격자 그룹 매핑을 생성하는 함수
def build_mapping(gg_map, min_width_ratio=0.4):
    regions, num = scipy.ndimage.label(gg_map+1) # 레이블링을 통해 연결된 구역을 식별
    min_width = gg_map.shape[1] * min_width_ratio # 최소 너비를 계산하기 위한 기준값 설정

    points = [] # 중심 좌표를 저장할 리스트
    coords_y = np.zeros_like(gg_map) # 격자 그룹의 중심 좌표를 저장할 배열
    period = 10 # 중심 좌표를 업데이트할 주기
    
    # 각 레이블별로 반복
    for i in range(num):
        y, x = np.where(regions==i+1) # 현재 레이블에 해당하는 좌표 추출
        w = np.max(x) - np.min(x) # 격자 그룹의 너비 계산
        if w < min_width:
            continue

        target_y = round(np.mean(y)) # 현재 레이블의 중심 y 좌표 계산

        uniq_x = np.unique(x) # 격자 그룹 내의 고유한 x 좌표 추출
        for ii, ux in enumerate(uniq_x):
            if ii % period == 0:
                meta_idx = np.where(x==ux)[0] # 현재 x 좌표에 해당하는 인덱스 추출
                sub_y = y[meta_idx] # 해당 x 좌표에 대한 y 좌표 추출
                cen_y = round(np.mean(sub_y)) # 해당 x 좌표에 대한 중심 y 좌표 계산
                coords_y[int(target_y), int(ux)] = cen_y # 격자 그룹의 중심 좌표 저장
                points.append((target_y, ux)) # 중심 좌표를 리스트에 추가

    # Add corner case
    coords_y[0] = 0
    coords_y[-1] = len(coords_y) - 1
    for i in range(coords_y.shape[1]):
        points.append((0, i))
        points.append((coords_y.shape[0]-1, i))

    return coords_y, np.array(points)


# 스태프 예측 이미지를 기반으로 좌표를 추정하는 함수
def estimate_coords(staff_pred):
    ker = np.ones((6, 1), dtype=np.uint8) # 침식 연산에 사용할 커널 생성
    pred = cv2.dilate(staff_pred.astype(np.uint8), ker) # 입력 스태프 이미지에 커널을 적용하여 팽창 연산 수행
    pred = morph_open(pred, (1, 6)) # 모폴로지 연산을 통해 이미지 개선

    logger.debug("Building grids")
    grid_map, grids = build_grid(pred)

    logger.debug("Labeling areas")
    gg_map, grid_groups = build_grid_group(grid_map, grids)

    logger.debug("Connecting lines")
    new_gg_map = connect_nearby_grid_group(gg_map, grid_groups, grid_map, grids)

    logger.debug("Building mapping")
    coords_y, points = build_mapping(new_gg_map)

    logger.debug("Dewarping")
    vals = coords_y[points[:, 0].astype(int), points[:, 1].astype(int)] # 좌표 정보를 기반으로 y 좌표값 추정
    grid_x, grid_y = np.mgrid[0:gg_map.shape[0]:1, 0:gg_map.shape[1]:1] # 그리드 생성
    coords_y = griddata(points, vals, (grid_x, grid_y), method='linear') # 그리드 기반으로 보간하여 y 좌표값 추정

    coords_x = grid_y.astype(np.float32) # 그리드의 x 좌표값
    coords_y = coords_y.astype(np.float32) # 보간된 y 좌표값
    return coords_x, coords_y


# 이미지를 왜곡 보정하는 함수
def dewarp(img, coords_x, coords_y):
    return cv2.remap(img.astype(np.float32), coords_x,coords_y, cv2.INTER_CUBIC)


if __name__ == "__main__":
    f_name = "wind2"
    #f_name = "last"
    #f_name = "tabi"
    img_path = f"../test_imgs/{f_name}.jpg"

    img_path = "../test_imgs/Chihiro/7.jpg"
    #img_path = "../test_imgs/Gym/2.jpg"

    ori_img = cv2.imread(img_path)
    f_name, ext = os.path.splitext(os.path.basename(img_path)) # 파일 이름과 확장자 추출
    parent_dir = os.path.dirname(img_path)
    pkl_path = os.path.join(parent_dir, f_name+".pkl")
    ff = pickle.load(open(pkl_path, "rb")) # pkl 파일 로드
    st_pred = ff['staff']
    ori_img = cv2.resize(ori_img, (st_pred.shape[1], st_pred.shape[0])) # 원본 이미지의 크기를 스태프 예측 결과에 맞게 조정

    ker = np.ones((6, 1), dtype=np.uint8) # 팽창 연산에 사용할 커널 생성
    pred = cv2.dilate(st_pred.astype(np.uint8), ker) # 팽창 연산을 통해 스태프 예측 결과 개선
    pred = morph_open(pred, (1, 6)) # 모폴로지 연산을 통해 이미지 개선

    print("Building grids")
    grid_map, grids = build_grid(pred)

    print("Labeling")
    gg_map, grid_groups = build_grid_group(grid_map, grids)

    print("Connecting")
    new_gg_map = connect_nearby_grid_group(gg_map, grid_groups, grid_map, grids)

    print("Estimating mapping")
    coords_y, points = build_mapping(new_gg_map)
    
    print("Dewarping")
    out = np.copy(ori_img)
    vals = coords_y[points[:, 0], points[:, 1]]
    grid_x, grid_y = np.mgrid[0:gg_map.shape[0]:1, 0:gg_map.shape[1]:1]
    mapping = griddata(points, vals, (grid_x, grid_y), method='linear')
    for i in range(out.shape[-1]):
        out[..., i] = cv2.remap(out[..., i].astype(np.float32), grid_y.astype(np.float32), mapping.astype(np.float32), cv2.INTER_CUBIC)


    mix = np.hstack([ori_img, out]) # 원본 이미지와 보정된 이미지를 가로로 결합


import random
# 이미지 처리 결과를 시각화하여 보여주는 함수
def teaser():
    # 스태프 예측 이미지
    plt.clf()
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.subplot(231)
    plt.title("Predict")
    plt.axis('off')
    plt.imshow(st_pred, cmap="Greys")

    # 변형된 이미지
    plt.subplot(232)
    plt.title("Morph")
    plt.axis('off')
    plt.imshow(pred, cmap='Greys')

    # 이진화된 이미지
    plt.subplot(233)
    plt.title("Quantize")
    plt.axis('off')
    plt.imshow(grid_map>0, cmap='Greys')

    # 격자 그룹을 다른 색으로 표시한 이미지
    plt.subplot(234)
    plt.title("Group")
    plt.axis('off')
    ggs = set(np.unique(gg_map))
    ggs.remove(-1)
    _gg_map = np.ones(gg_map.shape+(3,), dtype=np.uint8) * 255
    for i in ggs:
        ys, xs = np.where(gg_map==i)
        for c in range(3):
            v = random.randint(0, 255)
            _gg_map[ys, xs, c] = v
    plt.imshow(_gg_map)

    # 격자 그룹을 연결한 이미지
    plt.subplot(235)
    plt.title("Connect")
    plt.axis('off')
    plt.imshow(new_gg_map>0, cmap='Greys')

    # 왜곡 보정된 이미지
    plt.subplot(236)
    plt.title("Dewarp")
    plt.axis('off')
    plt.imshow(out)