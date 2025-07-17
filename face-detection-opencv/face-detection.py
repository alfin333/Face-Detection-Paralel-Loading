import cv2
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,640)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

progress = 0
max_progress = 100
step = 2

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        progress += step
        if progress > max_progress:
            progress = max_progress
    else:
        progress = 0

    for (x, y, w, h) in faces:
        # Kotak wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 243, 0), 2)

        # PROGRESS BAR
        bar_x = x
        bar_y = y + h + 10
        bar_w = w
        bar_h = 15

        # Hitung lebar progress saat ini
        filled_w = int((progress / max_progress) * bar_w)

        # Kotak background (abu-abu)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)

        # Kotak isi progress (biru)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (255, 255, 0), -1)

        # Tambah label persentase
        cv2.putText(frame, f'{progress}%', (bar_x + bar_w + 10, bar_y + bar_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 255), 1)

    cv2.imshow('Progress Bar Deteksi Wajah', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ketika semuanya selesai, capture release
cap.release()
cv2.destroyAllWindows()