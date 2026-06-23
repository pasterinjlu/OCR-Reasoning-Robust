"""Answer scoring: matcher -> refusal check -> LLM judge"""
from .matcher import extract_gt_answer, normalize_answer, check_answer
from .judge import judge_answer, is_refusal


def score_answer(gt, pred, question, question_type=None, use_judge=True):
    if not pred:
        return False
    if question_type == "multiple_choice":
        return check_answer(gt, pred, question_type="multiple_choice")
    if check_answer(gt, pred):
        return True
    if use_judge and not is_refusal(pred):
        return judge_answer(question, gt, pred)
    return False
