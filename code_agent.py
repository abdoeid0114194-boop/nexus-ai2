# code_agent.py: يكتب كود + اختبارات، يشغّله، ويصلحه لو فشل
import re, subprocess, tempfile, os
from model_engine import brain

LANGS = {  # الاسم: (امتداد, أمر التشغيل)
    "python": ("py", "python3 main.py"),
    "javascript": ("js", "node main.py"),
    "bash": ("sh", "bash main.py"),
    "c": ("c", "gcc main.py -o app -lm && ./app"),
    "c++": ("cpp", "g++ main.py -o app && ./app"),
    "java": ("java", "cp main.py Main.java && javac Main.java && java Main"),
}
DETECT = [("javascript", ["جافاسكريبت", "javascript", "node"]), ("c++", ["c++", "cpp", "سي بلس"]),
          ("java", ["جافا", "java"]), ("bash", ["bash", "باش"]), ("c", [" c ", "بلغة c", "سي "]),
          ("python", ["بايثون", "python"])]

def detect_lang(q):
    ql = " " + q.lower() + " "
    for lang, keys in DETECT:
        if any(k in ql for k in keys):
            return lang
    return "python"

def extract(text):
    m = re.search(r"```[\w+#]*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()

def run(lang, code, timeout=20):
    ext, cmd = LANGS[lang]
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "main.py"), "w", encoding="utf-8").write(code)
        try:
            p = subprocess.run(cmd, shell=True, cwd=d, capture_output=True, text=True, timeout=timeout)
            out = (p.stdout + p.stderr).strip()[-1500:]
            return p.returncode == 0 and "ALL TESTS PASSED" in out, out
        except subprocess.TimeoutExpired:
            return False, "انتهى الوقت (timeout)"

def code_task(q, tries=3):
    lang = detect_lang(q)
    sys = (f"You are an expert {lang} programmer. Write a complete, runnable {lang} program for the user's request, "
           "including automated tests (asserts) covering normal and edge cases. "
           "At the end it must print exactly ALL TESTS PASSED if every test passes. "
           "Output ONE code block only, no explanation.")
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": q}]
    code, out, ok = "", "", False
    for i in range(1, tries + 1):
        code = extract(brain.generate(msgs, max_new=900, temp=0.2, rep=1.05))
        ok, out = run(lang, code)
        if ok:
            break
        msgs += [{"role": "assistant", "content": f"```\n{code}\n```"},
                 {"role": "user", "content": f"Running it failed:\n{out}\nFix the code. Output ONE code block only."}]
    status = f"✅ الاختبارات نجحت (محاولة {i})" if ok else f"⚠️ لسه فيه مشكلة بعد {tries} محاولات"
    return f"```{lang}\n{code}\n```\n{status}\n\nناتج التشغيل:\n```\n{out}\n```"
