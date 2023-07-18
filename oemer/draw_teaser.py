from PIL import Image

import cv2
import numpy as np

from oemer import layers


def draw_bbox(bboxes, color, text=None, labels=None, text_y_pos=1):
    for idx, (x1, y1, x2, y2) in enumerate(bboxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        y_pos = y1 + round((y2-y1)*text_y_pos)
        if text is not None:
            cv2.putText(out, text, (x2+2, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        else:
            cv2.putText(out, labels[idx], (x2+2, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


# 다양한 악보 요소를 시각화하는 함수
def teaser() -> Image:
    ori_img = layers.get_layer('original_image') # 원본 이미지
    notes = layers.get_layer('notes') # 음표
    groups = layers.get_layer('note_groups') # 음표들이 그룹으로 이루어진 경우
    barlines = layers.get_layer('barlines') # 막대선
    clefs = layers.get_layer('clefs') # 조표
    sfns = layers.get_layer('sfns') # Staff에서 찾지 못한 음표
    rests = layers.get_layer('rests') # 쉼표

    global out
    out = np.copy(ori_img).astype(np.uint8) # 출력 이미지(out)를 원본 이미지로 초기화

    # 바운딩 박스 그리기
    draw_bbox([gg.bbox for gg in groups], color=(255, 192, 92), text="group") # 노란색
    draw_bbox([n.bbox for n in notes if not n.invalid], color=(194, 81, 167), labels=[str(n.label)[0] for n in notes if not n.invalid]) # 분홍색
    draw_bbox([b.bbox for b in barlines], color=(63, 87, 181), text='barline', text_y_pos=0.5) # 파란색
    draw_bbox([s.bbox for s in sfns if s.note_id is None], color=(90, 0, 168), labels=[str(s.label.name) for s in sfns if s.note_id is None]) # 보라색
    draw_bbox([c.bbox for c in clefs], color=(235, 64, 52), labels=[c.label.name for c in clefs]) # 빨간색
    draw_bbox([r.bbox for r in rests], color=(12, 145, 0), labels=[r.label.name for r in rests]) # 초록색

    for note in notes:
        if note.label is not None:
            x1, y1, x2, y2 = note.bbox
            cv2.putText(out, note.label.name[0], (x2+2, y2+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 70, 255), 2) # 음표의 라벨 그리기 (파란색)

    return Image.fromarray(out) # 출력 이미지를 PIL 이미지로 변환하여 반환
