import warnings; warnings.filterwarnings("ignore", ".*floordiv.*")
import os, sys, torch, pathlib
from datetime import datetime

LOG_FILE = pathlib.Path(__file__).parent / "history.txt"

def append_to_log(role: str, text: str):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{role}\t{text}\n")


SRC_DIR = pathlib.Path(__file__).parent / "src"
sys.path.append(str(SRC_DIR))

from src.transformers import (
    BlenderbotSmallTokenizer,
    BlenderbotSmallForConditionalGeneration,
)

CKPT   = "save/model/PERL-CAusion.ckpt"
BASE   = "blender"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = BlenderbotSmallTokenizer.from_pretrained(BASE)
model     = BlenderbotSmallForConditionalGeneration.from_pretrained(BASE)

tokenizer.sep_token = tokenizer.sep_token or tokenizer.eos_token
tokenizer.cls_token = tokenizer.cls_token or tokenizer.bos_token or tokenizer.eos_token

state = torch.load(CKPT, map_location="cpu")
state = state["model"] if isinstance(state, dict) and "model" in state else state
model.load_state_dict(state, strict=False)

if DEVICE.startswith("cuda"):
    model = model.half().to(DEVICE)
else:
    model = model.to(DEVICE)

model.eval()
print(f"\n🤖 Supporter ready on {DEVICE}!  (quit/exit 退出, /reset 清空历史)\n")

MAX_TURNS        = 8
MAX_MODEL_TOKENS = 512
MAX_GEN_NEW      = 128

GEN_KWARGS = dict(
    do_sample=True,
    top_p=0.97,
    temperature=0.95,
    repetition_penalty=1.5,
    no_repeat_ngram_size=5,
    max_length=None,          # 运行时再填
)

history = []

def build_prompt(hist):
    SEP = tokenizer.sep_token
    CLS = tokenizer.cls_token
    prompt = f"{CLS} " + f" {SEP} ".join(hist[-MAX_TURNS*2:]) + f" {SEP} supporter:"
    ids = tokenizer.encode(prompt)
    if len(ids) > MAX_MODEL_TOKENS - MAX_GEN_NEW:
        ids = ids[-(MAX_MODEL_TOKENS - MAX_GEN_NEW):]
        prompt = tokenizer.decode(ids, skip_special_tokens=False)
    return prompt

def save_session(hist, fn="chatlog.txt"):
    with open(fn, "a", encoding="utf-8") as f:
        f.write(f"\n=== {datetime.now():%F %T} ===\n")
        f.write("\n".join(hist) + "\n")
    print(f"💾 对话已保存到 {fn}")

while True:
    usr = input("You: ").strip()
    if usr.lower() in {"quit", "exit"}:
        break
    if usr.lower() == "/reset":
        history.clear()
        print("🔄 已清空历史\n")
        continue
    if usr.lower().startswith("/save"):
        save_session(history)
        continue

    history.append(f"seeker: {usr}")
    append_to_log("user", usr)

    prompt = build_prompt(history)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    GEN_KWARGS["max_length"] = inputs["input_ids"].shape[1] + MAX_GEN_NEW

    with torch.no_grad():
        outputs = model.generate(**inputs, **GEN_KWARGS,
                                 eos_token_id=tokenizer.eos_token_id)

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = reply.split("supporter:", 1)[-1].strip()

    print("Bot:", reply, "\n")
    history.append(f"supporter: {reply}")
    append_to_log("bot", reply)

    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()