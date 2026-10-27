"""
STEP 3: Real-time multi-person attendance using the trained ML model.

 - RED box   = faculty,  GREEN box = student (after faculty is verified)
 - YELLOW    = student seen but faculty NOT yet verified (attendance NOT marked)
 - ORANGE    = unknown person
 - Student attendance is marked only after the faculty face has been confirmed.
 - Many faces are handled in the same frame.
 - ONE csv file per day (attendance_YYYY-MM-DD.csv). Every take on the same day is
   stored in that same file; a new date automatically creates a new file.
 - The csv is saved when you press 'q'.

Run:  python attendance.py
"""
import os
import csv
from datetime import datetime
import cv2
import numpy as np
import joblib
import face_recognition
from sklearn.metrics import pairwise_distances

# ---------------- settings ----------------
MODEL_FILE = "model.pkl"
FACULTY_NAMES = ["amritha"]      # must match the folder name in dataset/raw
PROB_THRESHOLD = 0.60            # minimum model confidence
DIST_THRESHOLD = 0.55            # max distance to nearest known face (else Unknown)
CONFIRM_FRAMES = 3               # consecutive checks before attendance is marked
SCALE = 0.5                      # shrink frame for speed
PROCESS_EVERY = 2                # run recognition on every 2nd frame

RED, GREEN, YELLOW, ORANGE = (0, 0, 255), (0, 200, 0), (0, 255, 255), (0, 165, 255)

# ---------------- load model ----------------
bundle = joblib.load(MODEL_FILE)
model, classes, X_ref = bundle["model"], bundle["classes"], bundle["X_ref"]
print("Loaded model:", bundle.get("model_name", "ML model"), "| people:", classes)

for f in FACULTY_NAMES:
    if f not in classes:
        raise SystemExit(f"Faculty '{f}' not found in model classes {classes}")
students = [c for c in classes if c not in FACULTY_NAMES]

subject = input("Enter subject / class name: ").strip() or "N/A"
today = datetime.now().strftime("%Y-%m-%d")
CSV_FILE = f"attendance_{today}.csv"      # one file per day, in the project folder

attendance = {c: {"Role": "Faculty" if c in FACULTY_NAMES else "Student",
                  "Status": "Absent", "Time": "", "Conf": 0.0} for c in classes}
hits = {c: 0 for c in classes}
faculty_verified = False


def mark(name, conf):
    rec = attendance[name]
    if rec["Status"] == "Absent":
        rec["Status"] = "Present"
        rec["Time"] = datetime.now().strftime("%H:%M:%S")
        rec["Conf"] = round(conf * 100, 1)
        print(f"[{rec['Time']}] {name.title()} ({rec['Role']}) marked PRESENT")


# ---------------- CSV (one file per day) ----------------
HEADERS = ["Date", "Name", "Role", "Subject", "Status", "Time In", "Confidence (%)"]


def save_csv(path):
    """Create today's file if needed, otherwise update/append inside it."""
    rows = []
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = [r for r in csv.reader(f)][1:]          # skip header
    index = {(r[1], r[3]): i for i, r in enumerate(rows) if len(r) >= 7}   # (Name, Subject)

    for name, rec in attendance.items():
        new = [today, name.title(), rec["Role"], subject, rec["Status"], rec["Time"],
               rec["Conf"] if rec["Status"] == "Present" else ""]
        key = (name.title(), subject)
        if key in index:
            if rows[index[key]][4] == "Present":           # never downgrade an earlier Present
                continue
            rows[index[key]] = new
        else:
            rows.append(new)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig opens cleanly in Excel
        w = csv.writer(f)
        w.writerow(HEADERS)
        w.writerows(rows)


# ---------------- drawing helpers ----------------
def draw_label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - th - 10), (x + tw + 8, y), color, -1)
    cv2.putText(img, text, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_panel(img):
    lines = [(f"Subject: {subject}", (255, 255, 255)),
             ("Faculty: VERIFIED" if faculty_verified else "Faculty: NOT VERIFIED - waiting...",
              (0, 255, 0) if faculty_verified else (0, 0, 255))]
    for f in FACULTY_NAMES:
        s = attendance[f]
        lines.append((f"{f.title()} (Faculty): {s['Status']} {s['Time']}",
                      (0, 255, 0) if s["Status"] == "Present" else (200, 200, 200)))
    done = sum(attendance[s]["Status"] == "Present" for s in students)
    lines.append((f"Students present: {done}/{len(students)}", (255, 255, 255)))
    for s in students:
        r = attendance[s]
        lines.append((f"  {s.title()}: {r['Status']} {r['Time']}",
                      (0, 255, 0) if r["Status"] == "Present" else (200, 200, 200)))
    h = 12 + 24 * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (380, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, (t, col) in enumerate(lines):
        cv2.putText(img, t, (12, 28 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)


# ---------------- main loop ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_no, results = 0, []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        if frame_no % PROCESS_EVERY == 0:
            small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)

            results, seen = [], set()
            if encs:
                E = np.array(encs)
                probs = model.predict_proba(E)
                dists = pairwise_distances(E, X_ref).min(axis=1)
                for (t, r, b, l), p, d in zip(locs, probs, dists):
                    idx = int(np.argmax(p))
                    conf = float(p[idx])
                    name = classes[idx] if (conf >= PROB_THRESHOLD and d <= DIST_THRESHOLD) else "unknown"
                    box = (int(l / SCALE), int(t / SCALE), int(r / SCALE), int(b / SCALE))
                    results.append((box, name, conf))
                    if name != "unknown":
                        seen.add(name)

            for c in classes:
                hits[c] = hits[c] + 1 if c in seen else 0

            # 1) faculty first
            for f in FACULTY_NAMES:
                if hits[f] >= CONFIRM_FRAMES:
                    mark(f, max((c for _, n, c in results if n == f), default=1.0))
                    faculty_verified = True
            # 2) students only after faculty is verified
            if faculty_verified:
                for s in students:
                    if hits[s] >= CONFIRM_FRAMES:
                        mark(s, max((c for _, n, c in results if n == s), default=1.0))

        # draw boxes for every face in the frame
        for (x1, y1, x2, y2), name, conf in results:
            if name in FACULTY_NAMES:
                color, label = RED, f"{name.title()} (Faculty) {conf * 100:.0f}%"
            elif name == "unknown":
                color, label = ORANGE, "Unknown"
            elif faculty_verified:
                color, label = GREEN, f"{name.title()} {conf * 100:.0f}%"
            else:
                color, label = YELLOW, f"{name.title()} - faculty not verified"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, label, x1, y1, color)

        draw_panel(frame)
        cv2.imshow("Face Recognition Attendance (press q to save & quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    while True:
        try:
            save_csv(CSV_FILE)
            print("Saved", CSV_FILE)
            break
        except PermissionError:
            input(f"{CSV_FILE} is open in Excel. Close it and press Enter to save... ")
