from os import remove
import random
from pathlib import Path
from PIL import Image
import pickle

import tensorflow as tf
import augly.image as imaugs
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
# from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization

from oemer.bbox import get_bbox, merge_nearby_bbox, draw_bounding_boxes, rm_merge_overlap_bbox
from oemer.build_label import find_example


# SVM 분류기의 하이퍼파라미터 그리드 정의
SVM_PARAM_GRID = {
    'degree': [2, 3, 4],
    'decision_function_shape': ['ovo', 'ovr'],
    'C':[0.1, 1, 10, 100],
    'gamma':[0.0001, 0.001, 0.1, 1],
    'kernel':['rbf', 'poly']
}
# 이미지의 크기 정의
TARGET_WIDTH = 40
TARGET_HEIGHT = 70
DISTANCE = 10

# 데이터셋의 경로 정의
DATASET_PATH = "./ds2_dense/segmentation"


# 지정된 색상에 해당하는 데이터를 수집하여 저장하는 함수
def _collect(color, out_path, samples=5):
    out_path = Path(out_path)
    if not out_path.exists():
        out_path.mkdir()

    cur_samples = 0
    add_space = 10
    idx = 0
    while cur_samples < samples:
        # 해당 색상에 해당하는 데이터를 찾는다.
        arr = find_example(DATASET_PATH, color)
        if arr is None:
            continue
        arr[arr!=200] = 0
        # Bounding box를 얻어온다.
        boxes = get_bbox(arr)
        if len(boxes) > 1:
            boxes = merge_nearby_bbox(boxes, DISTANCE)
        boxes = rm_merge_overlap_bbox(boxes)
        # Bounding box를 활용하여 데이터를 추출하고 증강한다.
        for box in boxes:
            if idx >= samples:
                break
            print(f"{idx+1}/{samples}", end='\r')
            patch = arr[box[1]-add_space:box[3]+add_space, box[0]-add_space:box[2]+add_space]
            ratio = random.choice(np.arange(0.6, 1.3, 0.1))
            tar_w = int(ratio * patch.shape[1])
            tar_h = int(ratio * patch.shape[0])
            img = imaugs.resize(Image.fromarray(patch.astype(np.uint8)), width=tar_w, height=tar_h)

            seed = random.randint(0, 1000)
            img = imaugs.perspective_transform(img, seed=seed, sigma=3)
            img = np.where(np.array(img)>0, 255, 0)
            Image.fromarray(img.astype(np.uint8)).save(out_path / f"{idx}.png")
            idx += 1


        cur_samples += len(boxes)
    print()


# 데이터를 수집하여 train과 test 데이터를 생성하는 함수
def collect_data(samples=5):
    color_map = {
        # 74: "sharp",
        # 70: "flat",
        # 72: "natural",
        # 97: 'rest_whole',
        # 98: 'rest_half',
        # 99: 'rest_quarter',
        # 100: 'rest_8th',
        # 101: 'rest_16th',
        # 102: 'rest_32nd',
        # 103: 'rest_64th',
        # 10: 'gclef',
        # 13: 'fclef',
        # 추가: time signature에 해당하는 데이터
        # 21: 'timesig_0',
        # 22: 'timesig_1',
        # 23: 'timesig_2',
        24: 'timesig_3',
        25: 'timesig_4',
        26: 'timesig_5',
        27: 'timesig_6',
        # 28: 'timesig_7',
        # 29: 'timesig_8',
        # 30: 'timesig_9',
        # 33: 'timesig_4_4',
        # 34: 'timesig_2_2',
    }
    

    # color_map에 정의된 색상에 해당하는 데이터를 수집하여 저장
    for color, name in color_map.items():
        print('Current', name)
        _collect(color, f"train_data/{name}", samples=samples)
        _collect(color, f"test_data/{name}", samples=samples)


# 모델을 학습하는 함수
def train(folders):
    class_map = {idx: Path(ff).name for idx, ff in enumerate(folders)}
    train_x = []
    train_y = []
    samples = None
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        idx = 0
        for ff in folder.glob('*.png'):
            if samples is not None and idx >= samples:
                break
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img).flatten()
            train_x.append(arr)
            train_y.append(cidx)
            idx += 1

    print("Train model")
    # model = svm.SVC()#C=0.1, gamma=0.0001, kernel='poly', degree=2, decision_function_shape='ovo')
    #model = AdaBoostClassifier(n_estimators=50)
    #model = BaggingClassifier(n_estimators=50)  # For sfn classification
    #model = RandomForestClassifier(n_estimators=50)
    #model = GradientBoostingClassifier(n_estimators=50, verbose=1)
    model = GridSearchCV(svm.SVC(), SVM_PARAM_GRID)
    #model = KNeighborsClassifier(n_neighbors=len(folders))#, weights='distance')
    model.fit(train_x, train_y)
    return model, class_map


# TensorFlow 기반의 모델을 학습하는 함수
def train_tf(folders):
    class_map = {idx: Path(ff).name for idx, ff in enumerate(folders)}
    train_x = []
    train_y = []
    samples = None
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        idx = 0
        for ff in folder.iterdir():
            if samples is not None and idx >= samples:
                break
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img)
            train_x.append(arr)
            train_y.append(cidx)
            idx += 1
    train_x = np.array(train_x)[..., np.newaxis]
    train_y = tf.one_hot(train_y, len(folders))
    output_types = (tf.uint8, tf.uint8)
    output_shapes = ((TARGET_HEIGHT, TARGET_WIDTH, 1), (len(folders)))
    dataset = tf.data.Dataset.from_generator(lambda: zip(train_x, train_y), output_types=output_types, output_shapes=output_shapes)
    dataset = dataset.shuffle(len(train_y), reshuffle_each_iteration=True)
    dataset = dataset.repeat(5)
    dataset = dataset.batch(16)

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(TARGET_HEIGHT, TARGET_WIDTH, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(folders), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(dataset, batch_size=16, epochs=10)
    return model, class_map


# 테스트 데이터를 사용하여 모델을 평가하는 함수
def test(model, folders):
    test_x = []
    test_y = []
    samples = 5
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        idx = 0
        files = list(folder.glob('*.png'))
        random.shuffle(files)
        for ff in files:
            if idx >= samples:
                break
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img).flatten()
            test_x.append(arr)
            test_y.append(cidx)
            idx += 1

    pred_y = model.predict(test_x)
    tp_idx = (pred_y == test_y)
    tp = len(pred_y[tp_idx])
    acc = tp / len(test_y)
    print("Accuracy: ", acc)


# TensorFlow 기반의 모델을 평가하는 함수
def test_tf(model, folders):
    test_x = []
    test_y = []
    print("Loading data")
    for cidx, folder in enumerate(folders):
        folder = Path(folder)
        files = list(folder.iterdir())
        random.shuffle(files)
        for ff in files:
            img = Image.open(ff).resize((TARGET_WIDTH, TARGET_HEIGHT))
            arr = np.array(img)
            test_x.append(arr)
            test_y.append(cidx)

    test_x = np.array(test_x)[..., np.newaxis]
    test_y = np.array(test_y)
    test_result = []
    batch_size = 16
    for idx in range(0, len(test_x), batch_size):
        data = test_x[idx:idx+batch_size]
        pred = model.predict(data)
        pidx = np.argmax(pred, axis=-1)
        test_result.extend(list(pidx))

    test_result = np.array(test_result)
    tp = test_result[test_result==test_y].size
    acc = tp / len(test_y)
    print("Accuracy: ", acc)


# 이미지를 입력으로 받아 해당 이미지가 어떤 클래스에 해당하는지 예측하는 함수
def predict(region, model_name):
    if np.max(region) == 1:
        region *= 255
    # 저장된 학습된 모델 로드
    m_info = pickle.load(open(f"sklearn_models/{model_name}.model", "rb"))
    model = m_info['model']
    w = m_info['w']
    h = m_info['h']
    region = Image.fromarray(region.astype(np.uint8)).resize((w, h))
    pred = model.predict(np.array(region).reshape(1, -1))
    return m_info['class_map'][pred[0]]


if __name__ == "__main__":
    samples = 5
    # 데이터를 수집하여 train과 test 데이터 생성
    collect_data(samples=samples)

    # folders = ["gclef", "fclef"]; model_name = "clef"
    # folders = ["sharp", "flat", "natural"]; model_name = "sfn"
    # folders = ["rest_whole", "rest_quarter", "rest_8th"]; model_name = "rests"
    # folders = ["rest_8th", "rest_16th", "rest_32nd", "rest_64th"]; model_name = "rests_above8"
    # 박자표 데이터의 폴더와 모델 이름 지정
    folders = ["timesig_0", "timesig_1", "timesig_2", "timesig_3", "timesig_4", "timesig_5", "timesig_6", "timesig_7", "timesig_8", "timesig_9", "timesig_4_4", "timesig_2_2"]; model_name = "timesigs"

    #folders = ['clefs', 'sfns']; model_name = 'clefs_sfns'
    #folders = ["rest_whole", "rest_half", "rest_quarter", "rest_8th", "rest_16th", "rest_32nd", "rest_64th"]

    # Sklearn model
    model, class_map = train([f"ds2_dense/train_data/{folder}" for folder in folders])
    test(model, [f"ds2_dense/test_data/{folder}" for folder in folders])

    # TF-based model
    # model, class_map = train_tf([f"train_data/{folder}" for folder in folders])
    # test_tf(model, [f"test_data/{folder}" for folder in folders])

    # 학습된 모델 저장
    output = {'model': model, 'w': TARGET_WIDTH, 'h': TARGET_HEIGHT, 'class_map': class_map}
    pickle.dump(output, open(f"oemer/sklearn_models/{model_name}.model", "wb"))