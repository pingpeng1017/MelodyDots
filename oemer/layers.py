import numpy as np


# Dictionary of numpy.ndarray, representing all kinds of information
# extracted by different extractors.
_layers = {}

# 레이어에 접근한 횟수를 기록하는 딕셔너리
_access_count = {}


# 새로운 레이어를 등록하는 함수
def register_layer(name, layer):
    # 레이어 이름이 이미 등록되어 있는지 확인
    if name in _layers:
        print("Name already registered! Choose another name or delete it first.")
        return

    # 주어진 이름으로 레이어를 사전에 저장
    assert isinstance(layer, np.ndarray) # layer가 np.ndarray 타입인지 확인
    _layers[name] = layer # 'name'은 키(key)로 사용되며, 'layer'는 값(value)으로 저장됨
    _access_count[name] = 0 # 해당 레이어의 접근 횟수 초기화


# 등록된 레이어를 가져오는 함수
def get_layer(name):
    if name not in _layers:
        raise KeyError(f"The given layer name not registered: {name}")
    _access_count[name] += 1
    return _layers[name]


# 등록된 레이어를 삭제하는 함수
def delete_layer(name):
    if name in _layers:
        del _layers[name]
        del _access_count[name]


# 등록된 레이어의 이름을 리스트로 반환하는 함수
def list_layers():
    return list(_layers.keys())


# 레이어에 대한 접근 횟수를 출력하는 함수
def show_access_count():
    print(_access_count)