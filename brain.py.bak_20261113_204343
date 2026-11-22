
# brain.py: العقل. الملف الوحيد اللي بيعرف إن فيه Qwen.
# الاتفاق: generate(msgs, max_new, temp, rep, sample) -> str
# msgs = [{"role": "system"|"user"|"assistant", "content": "..."}]
import re, torch

BRAIN = "qwen"   # لما تدرّب موديلك غيّرها لـ "mine"

class QwenBrain:
    name = "Qwen2.5-7B-Instruct"

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        mn = "Qwen/Qwen2.5-7B-Instruct"
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        self.tok = AutoTokenizer.from_pretrained(mn)
        self.model = AutoModelForCausalLM.from_pretrained(mn, quantization_config=bnb, device_map="auto")
        cjk = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
        self.bad_ids = [[i] for t, i in self.tok.get_vocab().items()
                        if cjk.search(self.tok.convert_tokens_to_string([t]))]

    def generate(self, msgs, max_new=300, temp=0.3, rep=1.1, sample=True):
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        kw = dict(max_new_tokens=max_new, do_sample=sample)
        if sample:
            kw.update(temperature=temp, bad_words_ids=self.bad_ids, repetition_penalty=rep)
        out = self.model.generate(**inputs, **kw)
        return self.tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


class MyBrain:
    """عقلك اللي هتدرّبه. كمّل الدالتين لما يبقى جاهز."""
    name = "NEXUS-mine"

    def __init__(self):
        # حمّل موديلك هنا (من checkpoint على Drive)
        raise NotImplementedError("لسه مفيش موديل مدرّب")

    def generate(self, msgs, max_new=300, temp=0.3, rep=1.1, sample=True):
        # حوّل msgs لنص بالشكل اللي موديلك اتدرّب عليه، ولّد، وارجع النص الجديد بس
        raise NotImplementedError


BRAINS = {"qwen": QwenBrain, "mine": MyBrain}
brain = BRAINS[BRAIN]()
print("العقل الحالي:", brain.name)
