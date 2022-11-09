from sklearn.metrics import classification_report
import wandb


def get_classification_report(y_true, y_pred, class_labels, epoch=None):
    report = classification_report(
        y_true, y_pred, target_names=class_labels, output_dict=True
    )
    columns = ["epoch"] if epoch else []
    columns.extend(["class", "precision", "recall", "f1-score", "support"])

    data = []
    for cl in class_labels:
        row = [epoch, cl] if epoch else [cl]
        row.extend(
            [report[cl][k] for k in ["precision", "recall", "f1-score", "support"]]
        )
        data.append(row)

    return wandb.Table(
        data=data,
        columns=columns,
    )
