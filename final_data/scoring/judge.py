"""GPT-4o LLM-as-Judge for free-form answers"""
import os, time, requests

API_KEY = os.environ.get("OPENAI_API_KEY")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

REFUSAL_PATTERNS = [
    "cannot determine", "can't determine", "cannot be determined",
    "not possible to determine", "unable to determine",
    "not enough information", "insufficient information",
    "cannot answer", "can't answer", "unable to answer",
    "not provided", "not specified", "not given",
]


def is_refusal(pred):
    pred_lower = pred.lower().strip()
    return any(p in pred_lower for p in REFUSAL_PATTERNS)


def judge_answer(question, gt_answer, pred_answer, max_retries=3):
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set, but LLM-as-judge scoring was requested.")
    if is_refusal(pred_answer):
        return False
    prompt = f"""Judge whether the predicted answer is semantically equivalent to the ground truth answer.

Rules:
- Numbers must match exactly (5 ≠ 9, 3.7 ≠ 3.69)
- Minor formatting differences are OK ("$250" = "250", "3,000" = "3000")
- Refusals or "cannot determine" are NEVER correct

Question: {question}
Ground Truth: {gt_answer}
Prediction: {pred_answer}

Reply with only "Yes" or "No"."""

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}], "max_tokens": 8}

    for attempt in range(max_retries):
        try:
            resp = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=data, timeout=30)
            if resp.status_code == 200:
                reply = resp.json()["choices"][0]["message"]["content"].strip().lower()
                return reply.startswith("yes")
        except Exception as e:
            print(f"  Judge error (attempt {attempt+1}): {e}")
        time.sleep(1)
    return False
