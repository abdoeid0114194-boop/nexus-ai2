# model_engine.py: العقل. أي عقل لازم يحقق: name + generate(msgs, max_new, temp, rep, sample) -> str
import os, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class HFBrain:
    def __init__(self, path, name, lora=None):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(path)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb, device_map="auto")
        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora)
            self.name += "+LoRA"
        cjk = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
        self.bad_ids = [[i] for t, i in self.tok.get_vocab().items()
                        if cjk.search(self.tok.convert_tokens_to_string([t]))]

    def generate(self, msgs, max_new=300, temp=0.3, rep=1.1, sample=True):
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        kw = dict(max_new_tokens=max_new, do_sample=sample)
        if sample:
            kw.update(temperature=temp, bad_words_ids=self.bad_ids, repetition_penalty=rep)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **kw)
        return self.tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# لتركيب موديلك: اضبط NEXUS_BRAIN=mine و NEXUS_MODEL_PATH (أو NEXUS_LORA_PATH مع qwen)
BRAINS = {
    "qwen": lambda: HFBrain(os.getenv("NEXUS_QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                            "Qwen2.5-7B-Instruct", os.getenv("NEXUS_LORA_PATH")),
    "mine": lambda: HFBrain(os.environ["NEXUS_MODEL_PATH"], "NEXUS-mine"),
}

brain = BRAINS[os.getenv("NEXUS_BRAIN", "qwen")]()
print("العقل الحالي:", brain.name)
