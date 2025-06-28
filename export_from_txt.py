import pathlib
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

ROOT      = pathlib.Path(__file__).parent
TXT_FILE  = ROOT / "history.txt"         
TPL_DIR   = ROOT / "templates"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

def load_history(txt_path):
    hist = []
    with txt_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            role, text = line.rstrip("\n").split("\t", 1)
            hist.append((role, text))
    return hist

history = load_history(TXT_FILE)

env = Environment(loader=FileSystemLoader(TPL_DIR), autoescape=True)
tpl = env.get_template("chat.html")
html = tpl.render(
    title = f"情感支持对话系统",
    messages = history,
    typing=False,
)

ts   = datetime.now().strftime("%F_%H-%M-%S")
out  = REPORT_DIR / f"chat_{ts}.html"
out.write_text(html, encoding="utf-8")
print(f"✅ 已生成 {out.relative_to(ROOT)}")
