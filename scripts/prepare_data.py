import json
import os

def prepare_split(split_name):
    audio_dir = f"dataset/{split_name}/audio"
    text_dir = f"dataset/{split_name}/transcripts"
    output_file = f"data/{split_name}_all.json"

    data = []

    for fname in os.listdir(audio_dir):
        if not fname.endswith(".wav"):
            continue

        audio_path = os.path.join(audio_dir, fname)
        txt_path = os.path.join(text_dir, fname.replace(".wav", ".txt"))

        if not os.path.exists(txt_path):
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        data.append({
            "audio": audio_path,
            "text": text
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"{split_name}: {len(data)} samples saved")

prepare_split("train")
prepare_split("test")
