


def qa_open_ended(result):
    raise NotImplementedError()


def qa_multi_choice(result):
    answer_letter = result["answer_letter"]
    pred = result["pred"].strip()
    if ")" in pred:
        index = pred.index(")")
        pred_letter = pred[index - 1 : index]
    else:
        pred_letter = pred[0]

    if answer_letter.lower() == pred_letter.lower():
        return 1
    else:
        return 0


def scorer(task, results):
    total_score = 0
    for result in results:
        score = DATASET2METRIC[task](result)
        total_score += score

    return round(100 * total_score / len(results), 2)

    

DATASET2PROMPT = {
}

DATASET2MAXNEWTOKENS = {
    "qaego4d": 16,
    "cgbench": 16,
    "egoschema": 16,
    "mlvu": 16,
    "activitynet_qa": 1024,
}


DATASET2METRIC = {
    "qaego4d": qa_multi_choice,
    "cgbench": qa_multi_choice,
    "egoschema": qa_multi_choice,
    "mlvu": qa_multi_choice,
    "activitynet_qa": qa_open_ended,
}


DATASET2CATEGORY = {
    "qaego4d": "multiple_choice",
    "cgbench": "multiple_choice",
    "egoschema": "multiple_choice",
    "mlvu": "multiple_choice",
    "activitynet_qa": "open_ended"
}