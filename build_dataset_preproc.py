import json, pickle, pathlib, re, tqdm, argparse

def clean(text: str) -> str:
    """去掉 'seeker:' / 'supporter:' 前缀并做简单空白清洗"""
    return re.sub(r'^\s*(seeker|supporter)\s*:\s*', '', text, flags=re.I).strip()

def extract_samples(whole_dialog):
    """
    将一段 multi-turn 对话拆分为若干 (context, response) 样本。
    规则：每当出现 supporter 说话，就把之前所有 utterance 合成 context，
    该 supporter 语句作为 target。
    """
    samples = []
    for i, utt in enumerate(whole_dialog):
        if utt.lower().startswith("supporter:"):
            ctx  = " [SEP] ".join(map(clean, whole_dialog[:i])) or "[CLS]"
            tgt  = clean(utt)
            samples.append((ctx, tgt))
    return samples

def build(split_name, file_path):
    data = json.load(open(file_path, encoding="utf-8"))
    if isinstance(data, dict):                      # 你的 JSON 外层是 dict
        data = list(data.values())

    src, tgt, emo = [], [], []
    for dialog in tqdm.tqdm(data, desc=f"{split_name} dialogs"):
        samples = extract_samples(dialog["whole_dialog"])
        src.extend([c for c, _ in samples])
        tgt.extend([r for _, r in samples])
        # emotion_type 可选；如模型/代码不需要可忽略
        emo.extend([dialog["character"].get("emotion_type", "Other")] * len(samples))
    return {"source": src, "target": tgt, "emotion": emo}

def main(in_dir="data", out_path="data/dataset_preproc.p"):
    in_dir = pathlib.Path(in_dir)
    dataset = {
        "train": build("train", in_dir / "train.json"),
        "dev"  : build("val",   in_dir / "val.json"),
        "test" : build("test",  in_dir / "test.json"),
    }
    pickle.dump(dataset, open(out_path, "wb"))
    print(f"✅ Saved pickle to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="放 train/val/test.json 的文件夹")
    parser.add_argument("--out",      default=None,   help="输出 pkl 路径")
    args = parser.parse_args()
    out = args.out or f"{args.data_dir}/dataset_preproc.p"
    main(args.data_dir, out)
