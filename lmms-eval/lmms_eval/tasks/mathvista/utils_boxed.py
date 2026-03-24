from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content

from lmms_eval.tasks.mathvista import utils as _utils


def mathvista_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    text = _utils.mathvista_doc_to_text(doc, lmms_eval_specific_kwargs)
    return f"{text}\nPut your final answer in \\boxed{{}}."


def mathvista_process_results(doc, results):
    return _utils.mathvista_process_results(doc, [extract_final_boxed_content(results[0])])
