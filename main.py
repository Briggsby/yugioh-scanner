from functools import lru_cache
from pathlib import Path
import msvcrt
import re
import json
import urllib.request
import winsound

import cv2
import easyocr
import numpy as np

CARD_ID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
CARD_ID_PATTERN = re.compile(r"^(\w+)-([A-Z]{2})(\d{3})$")

# OCR misread corrections: maps characters that look similar
# Supports multi-character replacements (e.g. H -> 11)
LETTER_REPLACEMENTS = {
    "0": "O", "1": "I", "4": "A", "5": "S", "6": "E", "8": "B",
}
DIGIT_REPLACEMENTS = {
    "O": "0", "I": "1", "A": "4", "S": "5", "T": "1", "E": "3",
    "B": "8", "Z": "7", "C": "0", "H": "11",
}

OCR_SIMILAR = {
    frozenset("1IL"): 0.2,
    frozenset("0OQC"): 0.2,
    frozenset("5S"): 0.2,
    frozenset("8B"): 0.2,
    frozenset("6G"): 0.3,
    frozenset("2Z"): 0.3,
    frozenset("UV"): 0.3,
}
MAX_OCR_DISTANCE = 1.5

MOTION_THRESHOLD = 30.0
SETTLE_THRESHOLD = 2.0
STABLE_FRAMES_REQUIRED = 5

OUTPUT_FILE = "output.txt"
CARDS_DATA_FILE = Path(__file__).parent / "data" / "cards.json"
CARD_SETS_DATA_FILE = Path(__file__).parent / "data" / "cardsets.json"

reader = easyocr.Reader(["en"], gpu=True)


@lru_cache()
def get_valid_set_codes() -> list[str]:
    with open(CARD_SETS_DATA_FILE, "r") as f:
        sets = json.load(f)
    return {set["set_code"] for set in sets}


@lru_cache()
def cards_by_set_code() -> list[str]:
    with open(CARDS_DATA_FILE, "r") as f:
        cards = json.load(f)["data"]
    cards_by_set_code = {
        set["set_code"]: card for card in cards for set in card.get("card_sets", [])
    }
    return cards_by_set_code


def capture_frame(
    cap: cv2.VideoCapture,
    auto_focus: bool = True,
    focus_value: int = 50,
) -> cv2.typing.MatLike | None:
    if auto_focus:
        for _ in range(80):
            cap.read()
    else:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, focus_value)
        for _ in range(10):
            cap.read()

    ret, frame = cap.read()
    return frame if ret else None


def crop_card_id_region(
    frame: cv2.typing.MatLike,
    x_start: float = 0,
    x_end: float = 1,
    y_start: float = 0,
    y_end: float = 1,
) -> cv2.typing.MatLike:
    """Crop to the card ID region: roughly 2/3 down, right side of the card."""
    h, w = frame.shape[:2]
    y_start = int(h * y_start)
    y_end = int(h * y_end)
    x_start = int(w * x_start)
    x_end = int(w * x_end)
    return frame[y_start:y_end, x_start:x_end]


def read_card_id(frame: cv2.typing.MatLike) -> str | None:
    """Find and read the card ID from a frame in a single OCR pass."""
    results = reader.readtext(frame, allowlist=CARD_ID_CHARS)
    if not results:
        return None

    candidates = []
    all_fixed = []
    for bbox, text, conf in results:
        fixed = fix_card_id(text)
        all_fixed.append(fixed)
        if CARD_ID_PATTERN.match(fixed):
            candidates.append((bbox, fixed))

    if not candidates:
        best_raw = all_fixed[0] if all_fixed else "nothing detected"
        print(f"  No valid card ID found (best guess: {best_raw})")
        return None

    _, best_text = min(candidates, key=lambda r: min(p[1] for p in r[0]))
    return best_text


def ocr_sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0
    pair = frozenset((a, b))
    for group, cost in OCR_SIMILAR.items():
        if pair <= group:
            return cost
    return 1.0


def ocr_distance(s1: str, s2: str) -> float:
    m, n = len(s1), len(s2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + ocr_sub_cost(s1[i - 1], s2[j - 1]),
            )
    return dp[m][n]


def match_set_code(ocr_prefix: str) -> str:
    """Fuzzy-match an OCR'd prefix against valid set codes using OCR-aware distance."""
    valid = get_valid_set_codes()
    if ocr_prefix in valid:
        return ocr_prefix

    best_code = None
    best_dist = float("inf")
    for code in valid:
        dist = ocr_distance(ocr_prefix, code)
        if dist < best_dist:
            best_dist = dist
            best_code = code

    if best_code and best_dist <= MAX_OCR_DISTANCE:
        return best_code

    return best_code


def fix_card_id(raw: str) -> str:
    """Apply the known card id format to fix common OCR misreads."""
    raw = raw.upper().replace(" ", "")

    if "-" not in raw:
        for i, ch in enumerate(raw):
            if ch.isdigit():
                raw = raw[:i] + "-" + raw[i:]
                break

    parts = raw.split("-", 1)
    if len(parts) != 2:
        return raw

    prefix = parts[0]
    prefix = match_set_code(prefix)

    suffix = parts[1]
    if len(suffix) >= 2:
        lang = "".join(LETTER_REPLACEMENTS.get(c, c) for c in suffix[:2])
        num = "".join(DIGIT_REPLACEMENTS.get(c, c) for c in suffix[2:])
        suffix = lang + num

    return f"{prefix}-{suffix}"


def show_card_image(card: dict):
    """Download and display the card image in a window."""
    images = card.get("card_images", [])
    if not images:
        return
    url = images[0].get("image_url", "")
    if not url:
        return
    try:
        with urllib.request.urlopen(url) as resp:
            img_data = np.frombuffer(resp.read(), dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is not None:
            cv2.namedWindow("Last Scanned Card", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("Last Scanned Card", 500, 730)
            cv2.imshow("Last Scanned Card", img)
            cv2.waitKey(1)
    except Exception:
        pass


def remove_last_entry():
    """Remove the last line from the output file."""
    try:
        with open(OUTPUT_FILE, "r") as f:
            lines = f.readlines()
        if lines:
            removed = lines.pop().strip()
            with open(OUTPUT_FILE, "w") as f:
                f.writelines(lines)
            print(f"  Undone: {removed}")
            winsound.Beep(600, 100)
            winsound.Beep(400, 100)
        else:
            print("  Nothing to remove")
    except FileNotFoundError:
        print("  Nothing to remove")


def frame_diff(a: cv2.typing.MatLike, b: cv2.typing.MatLike) -> float:
    """Mean absolute difference between two frames (grayscale)."""
    g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(g1, g2)))


def main():
    cap = cv2.VideoCapture(2, cv2.CAP_MSMF)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    for _ in range(80):
        cap.read()

    print("Scanning... (Space = force scan, Backspace = undo last, Ctrl+C = stop)")

    _, prev_frame = cap.read()
    motion_detected = False
    stable_count = 0
    first_scan = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.waitKey(1)

            force_scan = False
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b" ":
                    force_scan = True
                elif key == b"\x08":  # Backspace
                    remove_last_entry()
                    continue

            should_read = force_scan or first_scan

            if not first_scan:
                diff = frame_diff(prev_frame, frame)
                prev_frame = frame

                if diff > MOTION_THRESHOLD:
                    motion_detected = True
                    stable_count = 0
                elif motion_detected:
                    if diff < SETTLE_THRESHOLD:
                        stable_count += 1
                    else:
                        stable_count = 0
                    if stable_count >= STABLE_FRAMES_REQUIRED:
                        should_read = True
                        motion_detected = False
                        stable_count = 0

            if should_read:
                first_scan = False
                card_id = read_card_id(frame)
                card = cards_by_set_code().get(card_id)
                if card:
                    print(f"  {card_id}: {card['name']}")
                    with open(OUTPUT_FILE, "a") as f:
                        f.write(card_id + "\n")
                    winsound.Beep(1000, 200)
                    show_card_image(card)
                else:
                    print(f"  No card found for {card_id}")
                    winsound.Beep(400, 400)
                print("  Ready.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
