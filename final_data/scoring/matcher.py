"""Answer matching and normalization"""
import re

MC_LETTER_RE = re.compile(r'\*{0,2}([A-Da-d])\*{0,2}[\.\)\s:]')
MC_ANSWER_RE = re.compile(r'(?:answer|option)\s*(?:is\s*[:.]?\s*|:\s*)\*{0,2}([A-Da-d])\*{0,2}[\.\)\s:\b]', re.IGNORECASE | re.DOTALL)


def extract_gt_answer(answer_text):
    for pattern in [r"The final answer is:\s*(.+?)(?:\n|$)", r"最终的回答是:\s*(.+?)(?:\n|$)"]:
        m = re.search(pattern, answer_text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def normalize_answer(answer):
    if answer is None:
        return ""
    answer = str(answer).strip()
    answer = re.sub(r'[,，。.、]$', '', answer)
    answer = answer.lower()
    if answer in ("yes", "true"):
        return "true"
    if answer in ("no", "false"):
        return "false"
    return answer


def normalize_numeric(s):
    s = re.sub(r'[$%€£¥]', '', s)
    s = re.sub(r'[元分角万亿千百十块]', '', s)
    s = re.sub(r'[,\s]', '', s)
    try:
        return float(s)
    except ValueError:
        return None


def extract_mc_letter(pred):
    pred = pred.strip()
    if len(pred) == 1 and pred.upper() in "ABCD":
        return pred.upper()
    m = MC_LETTER_RE.match(pred)
    if m:
        return m.group(1).upper()
    m2 = MC_ANSWER_RE.search(pred)
    if m2:
        return m2.group(1).upper()
    m3 = re.search(r'\*\*([A-Da-d])\*{0,2}(?:[\.\)\s:]|$)', pred)
    if m3:
        return m3.group(1).upper()
    return None


def check_answer_mc(gt, pred):
    gt_letter = gt.strip().upper()
    if gt_letter not in "ABCD":
        return False
    pred_letter = extract_mc_letter(pred)
    return pred_letter == gt_letter if pred_letter else False


def check_answer(gt, pred, question_type=None):
    if question_type == "multiple_choice":
        return check_answer_mc(gt, pred)
    gt_norm = normalize_answer(gt)
    pred_norm = normalize_answer(pred)
    if not gt_norm or not pred_norm:
        return False
    if gt_norm == pred_norm:
        return True
    gt_num = normalize_numeric(gt_norm)
    pred_num = normalize_numeric(pred_norm)
    if gt_num is not None and pred_num is not None:
        return gt_num == pred_num
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True
    return False
