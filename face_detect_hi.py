import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import time

# --- 1. 초기 설정 ---
# MediaPipe Face Mesh 설정 (눈동자 추적 포함)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # 눈동자 랜드마크를 위해 필수
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

# 분석 결과를 저장할 변수 (매 프레임 분석하면 느려지므로 캐싱용)
current_emotion = ""
current_gender = ""
current_age = ""
frame_count = 0
analysis_interval = 30  # 30프레임마다 속성(나이/성별/감정) 분석

def get_gaze_direction(eye_points, iris_point, frame_width, frame_height):
    """
    눈의 좌우 꼬리 좌표와 눈동자 중심 좌표를 비교하여 시선 방향 추정
    """
    # 좌표 변환 (정규화 좌표 -> 픽셀 좌표)
    eye_left = np.array([eye_points[0].x * frame_width, eye_points[0].y * frame_height])
    eye_right = np.array([eye_points[1].x * frame_width, eye_points[1].y * frame_height])
    iris_center = np.array([iris_point.x * frame_width, iris_point.y * frame_height])
    
    # 눈의 전체 폭과 높이 계산
    eye_width = np.linalg.norm(eye_left - eye_right)
    eye_center = (eye_left + eye_right) / 2
    
    # 수평 비율 (0에 가까우면 오른쪽(화면상), 1에 가까우면 왼쪽)
    # 이미지상 왼쪽이 실제 오른쪽 눈임 (거울 모드 고려 필요)
    
    # 간단한 벡터 계산을 통한 방향 판별
    dx = iris_center[0] - eye_center[0]
    dy = iris_center[1] - eye_center[1]
    
    direction_x = "Center"
    direction_y = "Center"
    
    # 임계값(Threshold)은 환경에 따라 조절 필요
    if dx < -eye_width * 0.15: direction_x = "Right" # 화면상 왼쪽 -> 사용자의 오른쪽
    elif dx > eye_width * 0.15: direction_x = "Left" # 화면상 오른쪽 -> 사용자의 왼쪽
    
    if dy < -2: direction_y = "Up"
    elif dy > 2: direction_y = "Down"

    return f"{direction_y}-{direction_x}" if direction_y != "Center" else direction_x

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        break

    # 성능 향상을 위해 이미지 쓰기 방지 설정 및 RGB 변환
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # 그리기 설정을 위해 다시 BGR로 변환
    image.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # --- 2. 시선 추적 (MediaPipe 사용) ---
            # 왼쪽 눈 랜드마크 인덱스: 33(안쪽), 133(바깥쪽), 468(눈동자)
            # 오른쪽 눈 랜드마크 인덱스: 362(안쪽), 263(바깥쪽), 473(눈동자)
            
            left_eye_indices = [face_landmarks.landmark[33], face_landmarks.landmark[133]]
            left_iris = face_landmarks.landmark[468]
            
            gaze_dir = get_gaze_direction(left_eye_indices, left_iris, w, h)

            # --- 3. 속성 분석 (DeepFace 사용) ---
            # DeepFace는 무거우므로 n프레임마다 실행하거나 별도 스레드로 처리 추천
            if frame_count % analysis_interval == 0:
                try:
                    # DeepFace 분석을 위해 얼굴 영역만 크롭해서 보내면 더 빠름 (여기선 전체 이미지 사용)
                    # enforce_detection=False는 얼굴을 못 찾아도 에러를 내지 않게 함
                    analysis = DeepFace.analyze(image, 
                                                actions=['age', 'gender', 'emotion'], 
                                                enforce_detection=False, 
                                                silent=True)
                    # DeepFace 결과는 리스트로 반환될 수 있음
                    result = analysis[0] if isinstance(analysis, list) else analysis
                    
                    current_age = result['age']
                    current_gender = result['dominant_gender']
                    current_emotion = result['dominant_emotion']
                except Exception as e:
                    print(f"분석 오류: {e}")

            # --- 4. 시각화 (Overlay) ---
            # 얼굴 영역 계산 (Bounding Box)
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                if y < y_min: y_min = y
                if y > y_max: y_max = y
            
            # 박스 그리기
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # 정보 텍스트 출력
            info_text = f"Sex: {current_gender}, Age: {current_age}, Emo: {current_emotion}"
            gaze_text = f"Gaze: {gaze_dir}"
            
            cv2.putText(image, info_text, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, gaze_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 눈동자 위치 시각화 (녹색 점)
            cx, cy = int(left_iris.x * w), int(left_iris.y * h)
            cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)

    frame_count += 1
    
    # 화면 출력
    cv2.imshow('Face Analysis Project', image)

    # ESC 키를 누르면 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()