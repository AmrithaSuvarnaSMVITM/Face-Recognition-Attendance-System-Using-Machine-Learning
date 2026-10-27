import cv2, os, sys

name = input("Enter person name (lowercase): ").strip().lower()
save_dir = f"dataset/raw/{name}"
os.makedirs(save_dir, exist_ok=True)
count = len(os.listdir(save_dir))

cap = cv2.VideoCapture(0)
print("Press 's' to save a photo, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, f"{name}: {count} saved", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f"{save_dir}/{name}_{count:03d}.jpg", frame)
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
