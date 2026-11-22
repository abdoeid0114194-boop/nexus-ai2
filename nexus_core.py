
# nexus_core.py: كل حاجة غير العقل. بيكلّم brain.generate بس.
import re, json, ast, operator, requests, torch, traceback, gc, glob
from datetime import datetime
from zoneinfo import ZoneInfo
import gradio as gr
from PIL import Image
from bs4 import BeautifulSoup
from ddgs import DDGS
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from model_engine import brain
from code_agent import code_task

BASE = "/content/drive/MyDrive/NEXUS_AI"
LOG_FILE = f"{BASE}/chat_log.jsonl"
MEM_FILE = f"{BASE}/memory.json"
IMG_DIR = f"{BASE}/images"
DOCS_DIR = f"{BASE}/docs"
HEADERS = {"User-Agent": "Mozilla/5.0 (NEXUS-AI learning project)"}

SYSTEM = """أنت NEXUS (بالإنجليزي، حروف كبيرة)، مساعد ذكي بيتكلم عربي مصري بسيط وواضح.
جاوب بالعربي فقط، ممنوع أي لغة تانية.
لما حد يسألك أنت مين، قول: "أنا NEXUS، مساعدك الذكي."
جاوب مباشرة وباختصار.
لو السؤال عن شخص أو رقم أو تاريخ ومفيش مصدر قدامك، قول "مش متأكد" ومتألفش معلومة.

معلومات عن المالك (قولها بس لو حد سألك مين صنعك أو مين مالكك):
- الاسم: عبدالرحمن عيد سلطان
- شاب مصري عنده 18 سنة
- هو اللي بنى NEXUS وبيطوره.
لا تقول أي معلومة تانية عنه، ولو حد سأل عن تفاصيل خاصة قول إنك مش مخوّل تشارك دي."""

def generate(msgs, max_new, temp=0.3, rep=1.1, sample=True):
    return brain.generate(msgs, max_new=max_new, temp=temp, rep=rep, sample=sample)

def to_text(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return to_text(c.get("text", ""))
    if isinstance(c, (list, tuple)):
        return "".join(to_text(p) for p in c)
    return str(c)

print("بحمّل موديل البحث في الملفات...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
RAG = {"chunks": [], "emb": None}

def chunk_text(text, size=700, overlap=100):
    text = re.sub(r"\s+", " ", text).strip()
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out

def build_index():
    chunks = []
    for path in sorted(glob.glob(f"{DOCS_DIR}/**/*", recursive=True)):
        name = path.split("/")[-1]
        try:
            if path.lower().endswith(".pdf"):
                for pn, page in enumerate(PdfReader(path).pages, 1):
                    for c in chunk_text(page.extract_text() or ""):
                        if len(c) > 40:
                            chunks.append((name, f"صفحة {pn}", c))
            elif path.lower().endswith((".txt", ".md")):
                for c in chunk_text(open(path, encoding="utf-8", errors="ignore").read()):
                    if len(c) > 40:
                        chunks.append((name, "", c))
        except Exception as e:
            print("تعذّر قراءة", name, e)
    RAG["chunks"] = chunks
    RAG["emb"] = embedder.encode([c[2] for c in chunks], convert_to_tensor=True, normalize_embeddings=True) if chunks else None
    print(f"اتفهرس {len(chunks)} مقطع من الملفات.")
    return len(chunks)

build_index()

def rag_search(q, k=4, min_score=0.35):
    if RAG["emb"] is None:
        return []
    qe = embedder.encode(q, convert_to_tensor=True, normalize_embeddings=True)
    scores = (RAG["emb"] @ qe).cpu()
    top = torch.topk(scores, min(k, len(scores)))
    return [RAG["chunks"][i] for s, i in zip(top.values.tolist(), top.indices.tolist()) if s >= min_score]

def load_mem():
    try:
        with open(MEM_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_mem(items):
    with open(MEM_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)

def memory_command(q):
    s = q.strip()
    if s.startswith("حدّث الملفات") or s.startswith("حدث الملفات"):
        return f"قريت الملفات تاني. عندي {build_index()} مقطع."
    if s.startswith("افتكر"):
        fact = s[len("افتكر"):].strip(" :،")
        if fact:
            items = load_mem()
            items.append(fact)
            save_mem(items)
            return f"تمام، هفتكر إن: {fact}"
    if "اللي فاكره عني" in s or "اللي فاكرو عني" in s:
        items = load_mem()
        return ("فاكر عنك:\n" + "\n".join(f"- {i}" for i in items)) if items else "مفيش حاجة محفوظة عنك لسه."
    if s.startswith("انسى كل"):
        save_mem([])
        return "مسحت كل حاجة."
    return None

DAYS = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]

def tool_datetime():
    n = datetime.now(ZoneInfo("Africa/Cairo"))
    return f"النهاردة {DAYS[n.weekday()]} {n.strftime('%Y-%m-%d')} والساعة {n.strftime('%H:%M')} بتوقيت القاهرة."

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
       ast.USub: operator.neg, ast.FloorDiv: operator.floordiv}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("تعبير غير مسموح")

def tool_calc(q):
    q = q.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    msgs = [{"role": "system", "content": "حوّل المسألة الحسابية لتعبير بايثون واحد فقط (أرقام وعمليات + - * / ** % وأقواس). اكتب التعبير بس من غير أي كلام تاني."},
            {"role": "user", "content": q}]
    expr = generate(msgs, 40, sample=False).strip().strip("`").replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = expr.splitlines()[0] if expr else expr
    try:
        val = _eval(ast.parse(expr, mode="eval").body)
        return f"التعبير: {expr}\nالناتج: {val}"
    except Exception as e:
        return f"معرفتش أحسب ({e}). التعبير اللي طلع: {expr}"

def read_page(url, limit=3000):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header", "aside"]):
            t.decompose()
        return re.sub(r"\s+", " ", soup.get_text(" ")).strip()[:limit]
    except Exception as e:
        print("خطأ قراءة صفحة:", url, e)
        return ""

def wiki_search(q, lang="ar", n=2, limit=3000):
    try:
        base = f"https://{lang}.wikipedia.org/w/api.php"
        r = requests.get(base, headers=HEADERS, timeout=10, params={
            "action": "query", "list": "search", "srsearch": q, "srlimit": n, "format": "json"}).json()
        out = []
        for item in r.get("query", {}).get("search", []):
            title = item["title"]
            ex = requests.get(base, headers=HEADERS, timeout=10, params={
                "action": "query", "prop": "extracts", "explaintext": 1,
                "titles": title, "format": "json"}).json()
            page = next(iter(ex["query"]["pages"].values()))
            text = (page.get("extract") or "")[:limit]
            if text:
                out.append((f"ويكيبيديا: {title}", text, f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"))
        return out
    except Exception as e:
        print("خطأ ويكيبيديا:", e)
        return []

def tool_search(q):
    year = datetime.now(ZoneInfo("Africa/Cairo")).year
    results = wiki_search(q)
    try:
        ddg = DDGS().text(f"{q} {year}", max_results=4)
    except Exception as e:
        print("خطأ DuckDuckGo:", e)
        ddg = []
    for i, r in enumerate(ddg):
        body = read_page(r["href"]) if i < 2 else ""
        results.append((r["title"], body or r["body"], r["href"]))
    return results

_t2i = None
_i2i = None

def get_pipes():
    global _t2i, _i2i
    if _t2i is None:
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
        print("بحمّل موديل الصور...")
        _t2i = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        _i2i = AutoPipelineForImage2Image.from_pipe(_t2i)
    return _t2i, _i2i

def free_gpu():
    _t2i.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

def to_english_prompt(q):
    msgs = [{"role": "system", "content": "Translate the user's image request into one short, vivid English image prompt (max 30 words). Output only the prompt, nothing else."},
            {"role": "user", "content": q}]
    return generate(msgs, 60, sample=False).strip().strip('"')

def save_img(img, tag):
    path = f"{IMG_DIR}/{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    img.save(path)
    return path

def tool_image(q):
    prompt = to_english_prompt(q)
    t2i, _ = get_pipes()
    try:
        t2i.to("cuda")
        img = t2i(prompt=prompt, num_inference_steps=4, guidance_scale=0.0,
                  height=512, width=512).images[0]
    finally:
        free_gpu()
    return save_img(img, "img"), prompt

def tool_edit(q, img_path, strength=0.6):
    prompt = to_english_prompt(q)
    t2i, i2i = get_pipes()
    src = Image.open(img_path).convert("RGB").resize((512, 512))
    try:
        t2i.to("cuda")
        img = i2i(prompt=prompt, image=src, strength=strength,
                  num_inference_steps=4, guidance_scale=0.0).images[0]
    finally:
        free_gpu()
    return save_img(img, "edit"), prompt

IMG_K = ["ارسم", "إرسم", "صورة ل", "صوره ل", "اعمل صورة", "اعمل صوره", "ولّد صورة", "ولد صورة", "generate image", "draw "]
TIME_K = ["الساعة كام", "الساعه كام", "النهاردة كام", "النهارده كام", "تاريخ النهاردة", "تاريخ النهارده", "النهاردة إيه", "النهارده ايه", "what time", "today's date"]
CALC_K = ["احسب", "كام ", "ناتج", "جذر", "×", "÷", "+", "*", "/"]
CHAT_ONLY = ["السلام عليكم", "صباح", "مساء", "إزيك", "ازيك", "شكرا", "شكراً", "إنت مين", "انت مين", "مين صنعك", "مين مالكك"]
PERSONAL_K = ["رياضتي", "اسمي", "عني", "بحب", "أنا ", "انا ", "هوايتي", "دراستي", "فاكر", "فاكرني", "مفضل", "المفضلة", "المفضله"]
SEARCH_K = ["مين", "من هو", "من هي", "أخبار", "اخبار", "سعر", "أحدث", "احدث", "آخر", "اخر", "الأخير", "الاخير",
            "نتيجة", "نتيجه", "فاز", "عدد", "متى", "امتى", "إمتى", "تاريخ", "عاصمة", "عاصمه", "who", "when", "price", "latest"]

CODE_K = ["اكتب كود", "اكتبلي كود", "اكتب برنامج", "اكتبلي برنامج", "اكتب دالة", "اكتب function", "write code", "write a program", "اختبر الكود"]

def route(q):
    if any(k in q for k in IMG_K):
        return "IMAGE"
    if any(k in q.lower() for k in CODE_K):
        return "CODE"
    if any(k in q for k in CHAT_ONLY) and len(q) < 30:
        return "NONE"
    if any(k in q for k in PERSONAL_K):
        return "NONE"
    if any(k in q for k in TIME_K):
        return "TIME"
    if re.search(r"[0-9٠-٩]", q) and any(k in q for k in CALC_K):
        return "CALC"
    if rag_search(q):
        return "DOCS"
    if any(k in q for k in SEARCH_K):
        return "SEARCH"
    msgs = [{"role": "system", "content": "أنت مصنّف. رد بكلمة واحدة بس من: TIME أو CALC أو SEARCH أو NONE."},
            {"role": "user", "content": ("TIME: سؤال عن الوقت أو التاريخ الحالي.\n"
                                         "CALC: مسألة حسابية.\n"
                                         "SEARCH: معلومة واقعية عن شخص أو حدث أو رقم أو أخبار.\n"
                                         "NONE: تحية أو دردشة أو شرح عام أو كتابة.\n\n" f"السؤال: {q}")}]
    ans = generate(msgs, 5, sample=False).upper()
    for t in ["TIME", "CALC", "SEARCH"]:
        if t in ans:
            return t
    return "NONE"

def log(q, a, tool):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(), "q": q, "a": a, "tool": tool,
                                "brain": brain.name}, ensure_ascii=False) + "\n")
    except Exception as e:
        print("خطأ في التسجيل:", e)

def get_files(message):
    if isinstance(message, dict):
        out = []
        for f in (message.get("files", []) or []):
            out.append(f if isinstance(f, str) else f.get("path", ""))
        return [f for f in out if f]
    return []

def nexus_reply(message, history):
    try:
        q = to_text(message)
        img_files = get_files(message)

        if img_files and q.strip():
            path, prompt = tool_edit(q, img_files[0])
            log(q, f"EDIT: {prompt}", "EDIT")
            return [gr.Image(path), f"عدّلتها. الوصف: {prompt}"]
        if img_files:
            return "استلمت الصورة. اكتب وصف التعديل (مثلًا: خلّي الخلفية صحرا)."

        mem_reply = memory_command(q)
        if mem_reply:
            log(q, mem_reply, "MEMORY")
            return mem_reply

        tool = route(q)

        if tool == "IMAGE":
            path, prompt = tool_image(q)
            log(q, f"IMAGE: {prompt}", tool)
            return [gr.Image(path), f"رسمتها. الوصف: {prompt}"]

        if tool == "CODE":
            ans = code_task(q)
            log(q, ans, tool)
            return ans

        sources, docs_refs = [], []
        user_msg = q
        if tool == "TIME":
            user_msg = f"سؤال المستخدم: {q}\n\nمعلومة من أداة الوقت: {tool_datetime()}\nجاوب بجملة واحدة بالعربي المصري بناءً عليها."
        elif tool == "CALC":
            user_msg = f"سؤال المستخدم: {q}\n\nنتيجة الحاسبة:\n{tool_calc(q)}\nقول الناتج بوضوح بجملة واحدة، وماتغيّرش الرقم."
        elif tool == "DOCS":
            hits = rag_search(q)
            ctx = "\n\n".join(f"[{i+1}] ({n} {p})\n{c}" for i, (n, p, c) in enumerate(hits))
            docs_refs = [f"[{i+1}] {n} {p}".strip() for i, (n, p, c) in enumerate(hits)]
            user_msg = (f"سؤال المستخدم: {q}\n\nمقاطع من ملفات المستخدم:\n{ctx}\n\n"
                        "جاوب بالاعتماد على المقاطع دي بس، بعربي مصري بسيط وواضح، واذكر رقم المقطع [1] بعد المعلومة. "
                        "لا تضف معلومة مش موجودة فيها. لو الإجابة مش فيها، قول: مش لاقيها في ملفاتك.")
        elif tool == "SEARCH":
            sources = tool_search(q)
            if sources:
                ctx = "\n\n".join(f"[{i+1}] {t}\n{b}" for i, (t, b, u) in enumerate(sources))
                user_msg = (f"{tool_datetime()}\nلو السؤال فيه 'الأخير' أو 'آخر'، اختار أحدث حدث حصل قبل تاريخ النهاردة.\n\n"
                            f"سؤال المستخدم: {q}\n\nنتايج بحث (نصوص صفحات):\n{ctx}\n\n"
                            "لخّص الإجابة في 2 إلى 4 جمل بعربي مصري بسيط من غير تكرار، واذكر رقم المصدر [1] بعد المعلومة. "
                            "لا تضف أي معلومة مش موجودة في النتايج. "
                            "لو الإجابة مش مكتوبة حرفيًا في النتايج، قول: ملقيتش إجابة مؤكدة.")

        mem = load_mem()
        sys_text = SYSTEM + ("\n\nمعلومات محفوظة عن المستخدم (استخدمها لو لها علاقة بالسؤال):\n" + "\n".join(f"- {m}" for m in mem) if mem else "")
        msgs = [{"role": "system", "content": sys_text}]
        for m in (history or [])[-6:]:
            if isinstance(m, dict):
                role, content = m.get("role", "user"), m.get("content")
                if role in ("user", "assistant") and isinstance(content, (str, list)):
                    c = to_text(content)
                    if c:
                        msgs.append({"role": role, "content": c})
        msgs.append({"role": "user", "content": user_msg})

        answer = generate(msgs, 350, 0.3, rep=1.15)
        if docs_refs:
            answer += "\n\nمن ملفاتك:\n" + "\n".join(docs_refs)
        if sources:
            answer += "\n\nالمصادر:\n" + "\n".join(f"[{i+1}] {u}" for i, (_, _, u) in enumerate(sources))
        if tool != "NONE":
            answer += f"\n\n🔧 الأداة: {tool}"
        log(q, answer, tool)
        return answer
    except Exception as e:
        traceback.print_exc()
        return f"حصل خطأ: {type(e).__name__}: {e}"

demo = gr.ChatInterface(fn=nexus_reply, multimodal=True, title="NEXUS",
                        description=f"العقل: {brain.name}")
demo.launch(share=True, debug=True)
