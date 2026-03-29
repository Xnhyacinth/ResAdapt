from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content

from lmms_eval.tasks.egoschema import utils as _utils


def egoschema_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    question = doc["question"]
    if "option" in doc:
        for op in doc["option"]:
            question += "\n" + op
        post_prompt = "\nAnswer with the option's letter from the given choices directly. Put your final answer in \\boxed{}."

    return f"{pre_prompt}{question}{post_prompt}"


def egoschema_process_results_generation(doc, result):
    return _utils.egoschema_process_results_generation(doc, [extract_final_boxed_content(result[0])])
