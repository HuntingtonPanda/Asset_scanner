from transformers import AutoProcessor, ShieldGemma2ForImageClassification
from PIL import Image
import requests
import torch
import os

model_id = "google/shieldgemma-2-4b-it"

#url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
#image = Image.open(requests.get(url, stream=True).raw)

# ---------- limits ----------
MAX_IMAGES = 100          # stop after this many images

# ------- model ------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ShieldGemma2ForImageClassification.from_pretrained(model_id).to(device).eval()
processor = AutoProcessor.from_pretrained(model_id)

def create_log(dir_name: str):
    doc_name = dir_name + "_result.txt"
    f = open(doc_name, "w")
    #f.write("poopies")
    return f

def folder_walker(dir: str):
    images_seen = 0
    global MAX_IMAGES
    #last = dir.split("/")[-1]
    lastp = os.path.basename(dir)
    #print(f"last: {last}, lastp: {lastp}")
    result_file = create_log(lastp)
    try:
        for filename in os.listdir(dir):
            path = os.path.join(dir, filename)
            if os.path.isfile(path):
                print(f"File found: {path}")
                probabilities = image_classifire(path)
                prediction = threshold(probabilities)
                
                header = f"File name: {filename}\n"
                result_file.write(header)
                prediction_text = f"Prediction: {prediction}\n"
                result_file.write(prediction_text)
                probabilities_text = f"Probabilities: {probabilities}\n\n"
                result_file.write(probabilities_text)
                
                images_seen += 1
            else:
                print("File not found!")
                
            if images_seen == MAX_IMAGES:
                break
    finally:
        result_file.close()
    
    """
    # Use this if you have a folder of folders and want to do them all at once
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            print(f"File found: {full_path}")
            print(f"path: {dirpath}, dir_name: {dirnames}, file_name: {filename}")
    """

def threshold(probabilities: tuple):
    result = "Safe"
    pornagraphic = False
    dangerous = False
    gory = False
    if probabilities[0][0] > 0.5:
        pornagraphic = True
    if probabilities[1][0] > 0.5:
        dangerous = True
    if probabilities[2][0] > 0.5:
        gory = True  

    issues = []

    if pornagraphic:
        issues.append("pornographic")
    if dangerous:
        issues.append("dangerous")
    if gory:
        issues.append("gory")
        
    if issues:
        result = "May contain: " + ", ".join(issues) + " material"
    
    return result
        

def image_classifire(image_path: str):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Skipping {image_path}: not a valid image ({e})")
        return None

    model_inputs = processor(images=[image], return_tensors="pt").to(device)

    with torch.inference_mode():
        scores = model(**model_inputs)

    return scores.probabilities

if __name__ == "__main__":
    folder_walker("C:/Users/cohun/Documents/ECENGR117/Asset_scanner/Gladihoppers")