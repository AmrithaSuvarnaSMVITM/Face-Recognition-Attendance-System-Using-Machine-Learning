"""
STEP 1: Clean the dataset, split REAL photos into train/test, augment ONLY the
training photos, convert every face to a 128-d encoding and save features.npz

 - Photos where no face can be found are moved automatically to dataset/rejected/<person>/
   (set FAILED_ACTION = "delete" to delete them, or "keep" to leave them alone).
 - The 80/20 split is done AFTER removing bad photos, so the test set is never shrunk by them.

Run:  python build_dataset.py
"""
import os, random, math, shutil
import cv2
import numpy as np
import face_recognition

random.seed(42)
np.random.seed(42)

RAW_DIR = "dataset/raw"                 # dataset/raw/<person>/*.jpg
UNKNOWN_DIR = "dataset/unknown_test"    # optional: photos of strangers (for Unknown testing)
REJECT_DIR = "dataset/rejected"         # failed photos are moved here
FAILED_ACTION = "move"                  # "move" | "delete" | "keep"
TEST_SIZE = 0.2                         # 80/20 split on usable REAL photos
TARGET_TRAIN_PER_PERSON = 60            # real + augmented training samples per person
MIN_USABLE = 20                         # warn if a person has fewer usable photos
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ---------------- augmentation ----------------
def augment(img):
    out = img.copy()
    out = cv2.convertScaleAbs(out, alpha=random.uniform(0.6, 1.4), beta=random.randint(-50, 50))
    gamma = random.uniform(0.6, 1.6)
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
    out = cv2.LUT(out, table)
    h, w = out.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-15, 15), 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.5:
        out = cv2.flip(out, 1)
    if random.random() < 0.3:
        out = cv2.GaussianBlur(out, (5, 5), 0)
    if random.random() < 0.3:
        noise = np.random.normal(0, 8, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


# ---------------- face encoding ----------------
def _detect(rgb):
    for up in (1, 2):
        locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=up)
        if locs:
            return locs
    return []


def get_encoding(bgr_img):
    """Encoding of the largest face in the image (None if no face found)."""
    if bgr_img is None:
        return None
    if bgr_img.ndim == 2:
        bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)
    elif bgr_img.shape[2] == 4:
        bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGRA2BGR)
    bgr_img = bgr_img.astype(np.uint8)

    rgb = np.ascontiguousarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    locs = _detect(rgb)
    if not locs:   # retry after fixing brightness/contrast
        yuv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(yuv[:, :, 0])
        eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        rgb = np.ascontiguousarray(cv2.cvtColor(eq, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        locs = _detect(rgb)
    if not locs:
        return None
    if len(locs) > 1:
        locs = [max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))]
    return face_recognition.face_encodings(rgb, locs)[0]


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


def handle_failed(path, person):
    """Move / delete / keep a photo in which no face was found."""
    if FAILED_ACTION == "delete":
        os.remove(path)
    elif FAILED_ACTION == "move":
        dest_dir = os.path.join(REJECT_DIR, person)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(path, os.path.join(dest_dir, os.path.basename(path)))


# ---------------- main ----------------
X_train, y_train, groups = [], [], []
X_test, y_test = [], []
group_id = 0
failed_log = []

for person in sorted(os.listdir(RAW_DIR)):
    folder = os.path.join(RAW_DIR, person)
    if not os.path.isdir(folder):
        continue

    # 1) encode every photo once, separate usable from failed
    good = []                                   # (path, encoding)
    failed = []
    for f in list_images(folder):
        path = os.path.join(folder, f)
        enc = get_encoding(cv2.imread(path))
        if enc is None:
            failed.append(path)
        else:
            good.append((path, enc))

    for path in failed:                         # clean up after reading is finished
        failed_log.append(path)
        handle_failed(path, person)

    # 2) split usable photos 80/20
    random.shuffle(good)
    n_test = max(1, round(len(good) * TEST_SIZE)) if len(good) >= 2 else 0
    test_items, train_items = good[:n_test], good[n_test:]

    # TEST: real photos only, never augmented
    for _, enc in test_items:
        X_test.append(enc)
        y_test.append(person)

    # TRAIN: real photo + augmented copies (copies share a group id so
    # cross-validation never splits copies of one photo across folds)
    aug_per_img = max(0, math.ceil(TARGET_TRAIN_PER_PERSON / max(1, len(train_items))) - 1)
    for path, enc in train_items:
        X_train.append(enc); y_train.append(person); groups.append(group_id)
        if aug_per_img:
            img = cv2.imread(path)
            for _ in range(aug_per_img):
                e = get_encoding(augment(img))
                if e is not None:
                    X_train.append(e); y_train.append(person); groups.append(group_id)
        group_id += 1

    warn = "   <-- WARNING: too few usable photos, retake some" if len(good) < MIN_USABLE else ""
    print(f"{person:10s} usable={len(good):3d}  failed={len(failed):3d}  "
          f"train={len(train_items):3d}  test={len(test_items):3d}  aug/photo={aug_per_img}{warn}")

# Optional unknown people (only used to test the 'Unknown' rejection)
X_unknown = []
if os.path.isdir(UNKNOWN_DIR):
    for root, _, fs in os.walk(UNKNOWN_DIR):
        for f in fs:
            if f.lower().endswith(IMG_EXT):
                enc = get_encoding(cv2.imread(os.path.join(root, f)))
                if enc is not None:
                    X_unknown.append(enc)
X_unknown = np.array(X_unknown) if X_unknown else np.empty((0, 128))

np.savez("features.npz",
         X_train=np.array(X_train), y_train=np.array(y_train), groups=np.array(groups),
         X_test=np.array(X_test), y_test=np.array(y_test), X_unknown=X_unknown)

with open("failed_images.txt", "w") as fh:
    fh.write("\n".join(failed_log))

print("\nTrain samples:", len(X_train), "| Test samples:", len(X_test),
      "| Unknown samples:", len(X_unknown))
print(f"Failed photos ({FAILED_ACTION}d): {len(failed_log)}  -> listed in failed_images.txt"
      if FAILED_ACTION != "keep" else f"Failed photos (kept): {len(failed_log)}")
print("Saved features.npz")
