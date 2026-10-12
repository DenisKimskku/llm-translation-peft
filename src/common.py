"""Shared helpers: prompt construction, chat formatting, IO."""
import json, os
from typing import Iterable, List, Dict, Any

BACKBONE = "Qwen/Qwen3-1.7B"
TARGET_NAME = {"en": "English", "ko": "Korean", "english": "English", "korean": "Korean"}

# Default instruction template given in the project spec (used for the baseline).
DEFAULT_TEMPLATE = (
    "Translate the following text into {target_language}.\n"
    "The input may contain multiple languages. Preserve the meaning of the entire text. "
    "Output only the translation.\n"
    "Text: {input_text}"
)

def norm_target(t: str) -> str:
    t = str(t).strip().lower()
    if t in ("en", "eng", "english"):
        return "en"
    if t in ("ko", "kor", "korean"):
        return "ko"
    raise ValueError(f"unknown target language: {t}")

def build_user_message(text: str, target: str, template: str = DEFAULT_TEMPLATE) -> str:
    return template.format(target_language=TARGET_NAME[norm_target(target)], input_text=text)

def build_prompt(tok, text: str, target: str, template: str = DEFAULT_TEMPLATE, system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": build_user_message(text, target, template)})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def postprocess(pred: str, target: str) -> str:
    """Deterministic output normalisation (part of the inference design, selected on the official validation set):
    NFC + strip; for Korean targets remove commas (Korean references in this task do not use them) and
    map full-width sentence punctuation to ASCII."""
    import re, unicodedata
    p = unicodedata.normalize("NFC", pred).strip()
    if norm_target(target) == "ko":
        p = re.sub(r"\s*[,\uff0c\u3001]\s*", " ", p)
        p = p.replace("\u3002", ".").replace("\uff1f", "?").replace("\uff01", "!")
        p = re.sub(r"\s+", " ", p).strip()
    return p
