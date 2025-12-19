import os, json, random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_coco_captions(captions_json_path):
    with open(captions_json_path, "r") as f:
        data = json.load(f)
    # image_id -> file_name
    id2file = {img["id"]: img["file_name"] for img in data["images"]}
    # image_id -> list[captions]
    caps = defaultdict(list)
    for ann in data["annotations"]:
        caps[ann["image_id"]].append(ann["caption"])
    return id2file, caps

@torch.no_grad()
def main():
    random.seed(42)

    coco_root = "data/coco2017"
    split = "train2017"  # or "val2017"
    captions_path = os.path.join(coco_root, "annotations", f"captions_{split}.json")
    images_dir = os.path.join(coco_root, split)

    out_dir = "data/clip_emb"
    os.makedirs(out_dir, exist_ok=True)

    # 1) load COCO captions
    id2file, caps = load_coco_captions(captions_path)
    image_ids = sorted(list(caps.keys()))

    # OPTIONAL: take a subset for quick iteration
    image_ids = image_ids[:20000]

    # 2) load CLIP
    device = get_device()
    print("Device:", device)
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name)

    # 3) compute embeddings
    img_embs = []
    txt_embs = []
    metas = []

    batch_imgs = []
    batch_txts = []
    batch_meta = []
    batch_size = 64

    def flush_batch():
        nonlocal batch_imgs, batch_txts, batch_meta, img_embs, txt_embs, metas
        if not batch_imgs:
            return

        inputs = proc(text=batch_txts, images=batch_imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model(**inputs)

        # CLIPModel gives pooled embeddings for image/text:
        # out.image_embeds: (B, D), out.text_embeds: (B, D)
        iemb = out.image_embeds
        temb = out.text_embeds

        # normalize (common practice for CLIP similarity space)
        iemb = iemb / iemb.norm(dim=-1, keepdim=True)
        temb = temb / temb.norm(dim=-1, keepdim=True)

        img_embs.append(iemb.detach().cpu().numpy().astype(np.float32))
        txt_embs.append(temb.detach().cpu().numpy().astype(np.float32))
        metas.extend(batch_meta)

        batch_imgs, batch_txts, batch_meta = [], [], []

    for image_id in tqdm(image_ids, desc=f"Embedding {split}"):
        file_name = id2file.get(image_id)
        if file_name is None:
            continue
        img_path = os.path.join(images_dir, file_name)
        if not os.path.exists(img_path):
            continue

        caption = random.choice(caps[image_id])  # one caption per image

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        batch_imgs.append(img)
        batch_txts.append(caption)
        batch_meta.append({"image_id": int(image_id), "file_name": file_name, "caption": caption})

        if len(batch_imgs) >= batch_size:
            flush_batch()

    flush_batch()

    img_arr = np.concatenate(img_embs, axis=0)
    txt_arr = np.concatenate(txt_embs, axis=0)

    # 4) save
    np.save(os.path.join(out_dir, f"img_emb_{split}.npy"), img_arr)
    np.save(os.path.join(out_dir, f"txt_emb_{split}.npy"), txt_arr)

    meta_path = os.path.join(out_dir, f"meta_{split}.jsonl")
    with open(meta_path, "w") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("Saved:")
    print(" ", os.path.join(out_dir, f"img_emb_{split}.npy"), img_arr.shape)
    print(" ", os.path.join(out_dir, f"txt_emb_{split}.npy"), txt_arr.shape)
    print(" ", meta_path, len(metas))

if __name__ == "__main__":
    main()
