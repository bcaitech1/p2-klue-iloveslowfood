from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def evaluate(y_true, y_pred, average: str = "macro") -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average=average
    )
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    return dict(accuracy=accuracy, f1=f1, precision=precision, recall=recall)


def compute_metrics(pred) -> dict:
    """모델의 성능 검증(accuracy)을 위한 함수. transformers의 Trainer를 활용 시 사용됨

    Args:
        pred ([type]): prediction과 ground truth 인스턴스를 지닌 객체

    Returns:
        dict: 검증 결과
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}
