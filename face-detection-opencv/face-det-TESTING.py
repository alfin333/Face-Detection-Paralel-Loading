import cv2
import math

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variabel untuk melacak progress tiap wajah
face_states = []
max_progress = 100
step = 2
threshold_distance = 50  # toleransi perbedaan posisi (px) untuk anggap wajah yang sama

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Buat list sementara untuk frame ini
    updated_states = []

    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 243, 0), 2)
        # Coba cocokan dengan wajah lama berdasarkan posisi
        matched = False
        for state in face_states:
            dist = euclidean_distance(center_x, center_y, state["x"], state["y"])
            if dist < threshold_distance:
                state["x"] = center_x
                state["y"] = center_y
                state["box_x"] = x
                state["box_y"] = y
                state["box_w"] = w
                state["box_h"] = h
                state["progress"] = min(state["progress"] + step, max_progress)
                updated_states.append(state)
                matched = True
                break

        if not matched:
            # Wajah baru → progress dari 0
            new_state = {
                "x": center_x,
                "y": center_y,
                "box_x": x,
                "box_y": y,
                "box_w": w,
                "box_h": h,
                "progress": 0
            }

            updated_states.append(new_state)

    face_states = updated_states  # hanya simpan wajah aktif saat ini

    # Gambar wajah dan progress-nya
    for state in (face_states):
        x = state["box_x"]
        y = state["box_y"]
        w = state["box_w"]
        h = state["box_h"]

        # Gambar progress bar
        bar_y = y + h + 10
        bar_width = w
        bar_height = 15
        filled_width = int((state["progress"] / max_progress) * bar_width)
        if (state["progress"] >= 100):

            cv2.putText(frame, "ORANG", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (200, 200, 200), 2)
        cv2.rectangle(frame, (x, bar_y), (x + filled_width, bar_y + bar_height), (255, 255, 0), -1)
        cv2.putText(frame, f'{state["progress"]}%', (int(x+w*0.45), int(bar_y + bar_height/1.2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Progress Per Wajah", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # print(updated_states)
        break

cap.release()
cv2.destroyAllWindows()
