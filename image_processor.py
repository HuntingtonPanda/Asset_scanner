# image_processor.py
# pip install -U transformers pillow tqdm python-dotenv torch

import os, json, csv, warnings
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, ShieldGemma2ForImageClassification
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="Some weights of ShieldGemma2ForImageClassification")

# ---------- env ----------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------- limits ----------
MAX_IMAGES = 10_000          # stop after this many images
images_seen = 0

# ---------- model ----------
MODEL_ID = "google/shieldgemma-2-4b-it"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if DEVICE == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN, use_fast=True)
model = ShieldGemma2ForImageClassification.from_pretrained(
    MODEL_ID, token=HF_TOKEN, torch_dtype=DTYPE
).to(DEVICE).eval()

# ---------- policies: order & thresholds ----------
# Empirically, most current checkpoints align with: ["sexual", "dangerous", "violence_gore"]
# If you ever see mismatches, flip POLICY_ORDER and re-run; we also dump raw rows for inspection.
POLICY_ORDER = ["sexually_explicit", "dangerous_content", "violence_gore"]
REPORT_LABELS = ["nudity", "weapons", "gory"]   # your preferred names matching the order above
THRESHOLDS    = {"nudity": 0.50, "weapons": 0.50, "gory": 0.50}

def extract_yes_probs(outputs):
    """
    Returns [p_yes_sexual, p_yes_dangerous, p_yes_violence] in 0..1.
    Handles both outputs.probabilities (preferred) and logits.
    """
    if getattr(outputs, "probabilities", None) is not None:
        # shape: (1, 3, 2) -> squeeze -> (3, 2)
        p = outputs.probabilities.squeeze(0)
    else:
        # shape: (1, 3, 2) -> squeeze -> (3, 2), softmax per-policy
        logits = outputs.logits.squeeze(0)
        p = torch.softmax(logits, dim=-1)
    yes = p[:, 1].detach().float().cpu().tolist()
    # Save a tiny debug file once so you can check raw rows if needed
    try:
        if not os.path.exists("policy_rows_seen.json"):
            with open("policy_rows_seen.json", "w", encoding="utf-8") as f:
                json.dump({"rows_yes_probs": yes, "order": POLICY_ORDER}, f, indent=2)
    except Exception:
        pass
    return yes  # list of 3 floats

def classify_image(image_path: str):
    global images_seen
    if images_seen >= MAX_IMAGES:
        return "BUDGET_EXHAUSTED"

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[skip] {image_path} ({e})")
        return None

    inputs = processor(images=[image], return_tensors="pt").to(DEVICE)
    images_seen += 1

    with torch.inference_mode():
        outputs = model(**inputs)

    yes_probs = extract_yes_probs(outputs)  # [sexual_yes, dangerous_yes, violence_yes]
    # map to your reporting keys in the same order
    conf = {REPORT_LABELS[i]: float(yes_probs[i]) for i in range(3)}
    bits = [1 if conf[k] >= THRESHOLDS[k] else 0 for k in REPORT_LABELS]
    illegal_flag = 1 if any(bits) else 0
    vector = [illegal_flag] + [conf[k] for k in REPORT_LABELS]  # [illegal, nudity, weapons, gory]
    return bits, conf, vector

def scan_folders(root_dir: str):
    unsafe_results = []
    all_scanned = []
    stopped_on = None

    for folder, _, files in os.walk(root_dir):
        for f in tqdm(files, desc=f"Scanning {folder}"):
            if not f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
                continue
            img_path = os.path.join(folder, f)
            res = classify_image(img_path)
            if res == "BUDGET_EXHAUSTED":
                stopped_on = os.path.abspath(img_path)
                return unsafe_results, all_scanned, stopped_on
            if not res:
                continue

            bits, conf, vector = res
            row = {
                "folder": os.path.basename(folder),
                "file": f,
                "path": os.path.abspath(img_path),
                "illegal_flag": vector[0],
                "conf_nudity":  conf["nudity"],
                "conf_weapons": conf["weapons"],
                "conf_gory":    conf["gory"],
            }
            all_scanned.append(row)
            if row["illegal_flag"] == 1:
                unsafe_results.append({
                    **row,
                    "vector": vector,  # [illegal, nudity, weapons, gory]
                })
    return unsafe_results, all_scanned, stopped_on

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path

def save_csv(rows, path):
    cols = ["folder", "file", "path", "illegal_flag", "conf_nudity", "conf_weapons", "conf_gory"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    return path

if __name__ == "__main__":
    root = Path(r"C:\Users\cohun\Documents\ECENGR117\Asset_scanner\Game1").resolve()
    print(f"Scanning: {root}")
    unsafe, all_rows, stopped = scan_folders(str(root))

    print("\n--- Unsafe Images Found ---")
    print(json.dumps(unsafe, indent=2))
    print(f"\nTotal unsafe images: {len(unsafe)}  |  Images scanned: {images_seen}/{MAX_IMAGES}")
    if stopped:
        print(f"\n⚠️ Max images reached. Stopped on: {stopped}")

    save_json(unsafe, "unsafe_results.json")
    save_csv(unsafe, "unsafe_results.csv")
    save_json(all_rows, "all_scanned.json")
    save_csv(all_rows, "all_scanned.csv")
    print("\nSaved: unsafe_results.json, unsafe_results.csv, all_scanned.json, all_scanned.csv")
