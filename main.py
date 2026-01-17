import cv2
import mediapipe as mp
import numpy as np
import time
from deepface import DeepFace
import threading

class RealTimeFaceSystem:
    def __init__(self):
        # 1. MediaPipe 얼굴 Mesh 초기화 (빠른 감지 및 시선 추적용)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True, # 홍채(Iris) 랜드마크 포함
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 2. 상태 변수들
        self.frame_count = 0
        self.analysis_interval = 60 # 30프레임마다 속성(나이/성별/감정) 분석
        self.last_attributes = {} # 마지막으로 분석된 속성 저장 (Face Index: Data)
        self.lock = threading.Lock() # 스레드 간 데이터 충돌 방지

    def get_gaze_direction(self, frame, face_landmarks, frame_w, frame_h):
        """
        눈동자(Iris)의 위치를 눈의 좌우 코너와 비교하여 시선 방향 결정
        """
        # 왼쪽 눈 랜드마크 인덱스 (MediaPipe 기준)
        # 왼쪽 눈꼬리: 33, 오른쪽 눈꼬리: 133, 왼쪽 홍채 중심: 468
        eye_indices = [[362, 263, 473], [33, 133, 468]]
        
        # 좌표 변환 함수
        def to_coord(idx):
            pt = face_landmarks.landmark[idx]
            return int(pt.x * frame_w), int(pt.y * frame_h)
        
        for indices in eye_indices:
            p1, p2, iris = [to_coord(idx) for idx in indices]
            eye_center_x = (p1[0] + p2[0]) // 2
            eye_center_y = (p1[1] + p2[1]) // 2
            dx = iris[0] - eye_center_x
            dy = iris[1] - eye_center_y

            scale = 8.0 
            
            # 끝점 계산
            end_x = int(iris[0] + dx * scale)
            end_y = int(iris[1] + dy * scale)

            # 화살표 그리기 (노란색, 두께 2)
            cv2.arrowedLine(frame, (iris[0], iris[1]), (end_x, end_y), (0, 0, 255), 2, tipLength=0.3)

        # 눈의 전체 가로 길이
        eye_width = p2[0] - p1[0]
        if eye_width == 0: return "Unknown"

        # 홍채 중심이 눈의 어디에 위치하는지 비율 계산 (0.0 ~ 1.0)
        # 0에 가까우면 오른쪽(화면상 왼쪽), 1에 가까우면 왼쪽(화면상 오른쪽)
        # 미러링된 화면 기준 사용자의 실제 시선 방향으로 변환 필요
        
        relative_pos = (iris[0] - p1[0]) / eye_width
        
        # 수직 방향 (간단화: 홍채 Y축 위치)
        # 눈꺼풀 상단: 386, 하단: 374
        top_lid = to_coord(159)
        bot_lid = to_coord(145)
        eye_height = bot_lid[1] - top_lid[1]
        relative_height = (iris[1] - top_lid[1]) / eye_height if eye_height > 0 else 0.5
        direction = ""
        
        # 수평 판별
        if relative_pos < 0.35: direction += "Right " # 화면상 왼쪽 -> 사용자 기준 오른쪽
        elif relative_pos > 0.65: direction += "Left " # 화면상 오른쪽 -> 사용자 기준 왼쪽
        
        # 수직 판별
        if relative_height < 0.35: direction += "Up"
        elif relative_height > 0.65: direction += "Down"
        
        if direction == "": direction = "Center"
        
        return direction.strip()

    def draw_gaze_vector(self, frame, face_landmarks, w, h):
        """
        [추가된 기능] 눈동자 움직임에 따른 벡터 시각화 (노란색 화살표)
        """
        # (홍채 인덱스, 왼쪽 코너, 오른쪽 코너)
        # 왼쪽 눈: 468, 33, 133 / 오른쪽 눈: 473, 362, 263
        eyes_indices = [
            (468, 33, 133), 
            (473, 362, 263)
        ]

        def to_coord(idx):
            pt = face_landmarks.landmark[idx]
            return int(pt.x * w), int(pt.y * h)

        for iris_idx, corner1_idx, corner2_idx in eyes_indices:
            iris = to_coord(iris_idx)
            c1 = to_coord(corner1_idx)
            c2 = to_coord(corner2_idx)

            # 눈의 중심점 계산 (양쪽 코너의 중간)
            eye_center_x = (c1[0] + c2[0]) // 2
            eye_center_y = (c1[1] + c2[1]) // 2

            # 벡터 계산: 홍채 위치 - 눈 중심
            # dx, dy는 눈동자가 중심에서 얼마나 벗어났는지를 나타냄
            dx = iris[0] - eye_center_x
            dy = iris[1] - eye_center_y

            # 벡터 증폭 (화면에서 잘 보이도록 길이 늘리기)
            scale = 8.0 
            
            # 끝점 계산
            end_x = int(iris[0] + dx * scale)
            end_y = int(iris[1] + dy * scale)

            # 화살표 그리기 (노란색, 두께 2)
            cv2.arrowedLine(frame, (iris[0], iris[1]), (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)

    def analyze_attributes_thread(self, face_roi, idx):
        """
        별도 스레드에서 무거운 딥러닝 분석 수행
        """
        try:
            # DeepFace 분석 (가장 가벼운 모델 설정, detector는 이미 했으므로 skip)
            # 주의: enforce_detection=False로 해야 ROI만 넘길 때 에러 안 남
            objs = DeepFace.analyze(
                img_path=face_roi, 
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,
                detector_backend='skip',
                silent=True
            )
            
            result = objs[0]
            with self.lock:
                self.last_attributes[idx] = {
                    "age": result['age'],
                    "gender": max(result['gender'], key=result['gender'].get), # Male/Female
                    "emotion": result['dominant_emotion']
                }
        except Exception as e:
            # 분석 실패 시 (얼굴이 너무 흐릿하거나 작음)
            pass

    def process_stream(self):
        cap = cv2.VideoCapture(0) # 웹캠 0번
        prev_time = 0

        print("[INFO] 스트림 시작... (초기 모델 로딩에 시간이 걸릴 수 있습니다)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 좌우 반전 (거울 모드)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. 얼굴 Mesh 감지 (MediaPipe) - 매 프레임 수행 (가벼움)
            results = self.face_mesh.process(rgb_frame)

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time

            if results.multi_face_landmarks:
                for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    # Bounding Box 계산
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                    for lm in face_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                        if x < x_min: x_min = x
                        if x > x_max: x_max = x
                        if y < y_min: y_min = y
                        if y > y_max: y_max = y
                    
                    # 여유 공간 추가
                    margin = 20
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(w, x_max + margin)
                    y_max = min(h, y_max + margin)

                    # 2. 시선 추적 (매 프레임)
                    gaze = self.get_gaze_direction(frame,face_landmarks, w, h)

                    # 3. 속성 분석 (30 프레임마다 스레드로 실행)
                    if self.frame_count % self.analysis_interval == 0:
                        face_roi = rgb_frame[y_min:y_max, x_min:x_max]
                        if face_roi.size > 0:
                            # 비동기 실행 (메인 루프 멈춤 방지)
                            self.analyze_attributes_thread(face_roi, idx)

                    # 4. 정보 오버레이
                    # 박스 그리기
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # 시선 정보 표시
                    cv2.putText(frame, f"Gaze: {gaze}", (x_min, y_min - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # 속성 정보 표시 (저장된 값 사용)
                    if idx in self.last_attributes:
                        attr = self.last_attributes[idx]
                        info_str = f"{attr['gender']}, {attr['age']}, {attr['emotion']}"
                        cv2.putText(frame, info_str, (x_min, y_min - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, "Analyzing...", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            self.frame_count += 1
            
            # FPS 표시
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Smart Face Analysis', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = RealTimeFaceSystem()
    system.process_stream()
    system.process_stream()
