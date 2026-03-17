# ============================ PART 1 ============================

from config import HF_API_KEY
import requests
import base64
import os
import re
from PIL import Image
from colorama import init, Fore, Style

init(autoreset=True)

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
}

VISION_MODELS = [
    "Qwen/Qwen3-VL-8B-Instruct:together",
    "Qwen/Qwen3-VL-32B-Instruct:together",
    "Qwen/Qwen2.5-VL-7B-Instruct:together",
    "Qwen/Qwen2.5-VL-32B-Instruct:together",
    "Qwen/Qwen2-VL-2B-Instruct:together",
    "Qwen/Qwen2-VL-7B-Instruct:together",
]

TEXT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct:together",
    "Qwen/Qwen2.5-14B-Instruct:together",
    "Qwen/Qwen2.5-32B-Instruct:together",
    "mistralai/Mistral-7B-Instruct-v0.3:together",
    "mistralai/Mixtral-8x7B-Instruct-v0.1:together",
]


def _data_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")


def query_hf_api(payload: dict):
    try:
        r = requests.post(ROUTER_URL, headers=HEADERS, json=payload, timeout=120)
    except requests.RequestException as e:
        return None, f"Request failed: {e}"

    if r.status_code != 200:
        try:
            j = r.json()
            msg = j.get("error", {}).get("message") or str(j)
        except Exception:
            msg = (r.text or "").strip() or r.reason or "Request failed."
        return None, f"Status {r.status_code}: {msg}"

    try:
        return r.json(), None
    except Exception:
        return None, "Non-JSON response received from the API."


def _extract_text(data) -> str:
    msg = (data or {}).get("choices", [{}])[0].get("message", {}) or {}
    return (msg.get("content") or "").strip()


def _run_models(models, messages, max_tokens=160, temperature=0.5):
    last_err = None

    for model in models:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        data, err = query_hf_api(payload)

        if err:
            last_err = err
            continue

        out = _extract_text(data)
        if out:
            return out, None

        last_err = "Empty response from model."

    return None, last_err or "All models failed."


def _words(text: str):
    return re.findall(r"\S+", (text or "").strip())


def _ensure_sentence_end(text: str) -> str:
    t = (text or "").strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t


# ============================ TEXT GENERATION ============================

def generate_text(prompt: str, max_new_tokens: int = 220) -> str:
    messages = [{"role": "user", "content": prompt}]
    out, err = _run_models(TEXT_MODELS, messages, max_tokens=max_new_tokens, temperature=0.6)

    if err:
        raise Exception(err)

    if out is None:
        raise Exception("Model returned no output.")

    return out

def generate_exact_sentence(prompt: str, n_words: int, max_new_tokens: int) -> str:
    # Force overshoot so trimming always works
    expanded_prompt = (
        f"Write a detailed descriptive paragraph of at least {n_words + 25} words. "
        "Use exactly one complete sentence and end with a period.\n\n"
        + prompt
    )

    out = generate_text(expanded_prompt, max_new_tokens=max_new_tokens)
    words = _words(out)

    if len(words) < n_words:
        words += ["clearly"] * (n_words - len(words))

    trimmed = " ".join(words[:n_words])
    return _ensure_sentence_end(trimmed)


# ============================ IMAGE CAPTION ============================

def get_basic_caption(image_path: str) -> str:
    print(f"{Fore.YELLOW}Generating basic caption...")

    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Write one complete sentence describing this image."},
            {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
        ],
    }]

    cap, err = _run_models(VISION_MODELS, msgs, max_tokens=90, temperature=0.2)
    return cap if cap else f"[Error] {err}"


def print_menu():
    print(f"""{Style.BRIGHT}{Fore.GREEN}
================ Image-to-Text Conversion =================
1. Caption (5 words)
2. Description (30 words)
3. Summary (50 words)
4. Exit
============================================================
""")


def main():
    image_path = input(f"{Fore.BLUE}Enter image path (e.g., test.jpg): {Style.RESET_ALL}")

    if not os.path.exists(image_path):
        print(f"{Fore.RED}File '{image_path}' does not exist.")
        return

    try:
        Image.open(image_path)
    except Exception as e:
        print(f"{Fore.RED}Failed to open image: {e}")
        return

    basic_caption = get_basic_caption(image_path)
    print(f"{Fore.YELLOW}Basic caption: {Style.BRIGHT}{basic_caption}\n")

    while True:
        print_menu()
        choice = input(f"{Fore.CYAN}Enter choice (1-4): {Style.RESET_ALL}").strip()

        if choice == "1":
            out = " ".join(_words(basic_caption)[:5])
            print(f"{Fore.GREEN}Caption (5 words): {Fore.YELLOW}{Style.BRIGHT}{_ensure_sentence_end(out)}\n")

        elif choice == "2":
            prompt = (
                "Expand this into a detailed descriptive sentence.\n\n"
                "Text: " + basic_caption
            )
            out = generate_exact_sentence(prompt, 30, max_new_tokens=220)
            print(f"{Fore.GREEN}Description (30 words): {Fore.YELLOW}{Style.BRIGHT}{out}\n")

        elif choice == "3":
            prompt = (
                "Provide a rich, detailed summary of the scene shown below.\n\n"
                "Image seed: " + basic_caption
            )
            out = generate_exact_sentence(prompt, 50, max_new_tokens=280)
            print(f"{Fore.GREEN}Summary (50 words): {Fore.YELLOW}{Style.BRIGHT}{out}\n")

        elif choice == "4":
            print(f"{Fore.GREEN}Goodbye!")
            break

        else:
            print(f"{Fore.RED}Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()