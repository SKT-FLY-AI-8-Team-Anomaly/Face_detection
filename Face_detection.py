import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3개 모델 로드
AGE_MODEL = "Newvel/face_age_detection_base_v3_weighted"
GENDER_MODEL = "dima806/man_woman_face_image_detection"
EMOTION_MODEL = "abhilash88/face-emotion-detection"

def load_clf(repo_id):
    processor = AutoImageProcessor.from_pretrained(repo_id)
    model = AutoModelForImageClassification.from_pretrained(repo_id).to(DEVICE).eval()
    return processor, model

age_proc, age_model = load_clf(AGE_MODEL)
gen_proc, gen_model = load_clf(GENDER_MODEL)
emo_proc, emo_model = load_clf(EMOTION_MODEL)

@torch.inference_mode()
def predict_one(pil_img: Image.Image, proc, model):
    inputs = proc(images=pil_img, return_tensors="pt").to(DEVICE)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs).item())
    score = float(probs[idx].item())
    label = model.config.id2label.get(idx, str(idx))
    return label, score

# 얼굴 검출기(가벼움)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    for (x, y, fw, fh) in faces:
        # 약간 여유를 줘서 크롭
        pad = int(0.15 * max(fw, fh))
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(w, x + fw + pad); y2 = min(h, y + fh + pad)

        face_bgr = frame[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(face_rgb)

        age_label, age_score = predict_one(pil, age_proc, age_model)
        gen_label, gen_score = predict_one(pil, gen_proc, gen_model)
        emo_label, emo_score = predict_one(pil, emo_proc, emo_model)

        # 박스 + 텍스트
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lines = [
            f"Age: {age_label} ({age_score:.2f})",
            f"Gender: {gen_label} ({gen_score:.2f})",
            f"Emotion: {emo_label} ({emo_score:.2f})",
        ]
        ty = y1 - 10
        for s in lines[::-1]:
            cv2.putText(frame, s, (x1, max(20, ty)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            ty -= 22

    cv2.imshow("Age / Gender / Emotion (Realtime)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
