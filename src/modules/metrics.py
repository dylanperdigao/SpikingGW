import pandas as pd
import numpy as np
from aequitas.group import Group    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, auc, roc_curve, precision_recall_curve, roc_auc_score

def metrics_performance(y_true, y_prob):
    fprs, tprs, thresholds = roc_curve(y_true, y_prob)   
    threshold = np.min(thresholds[fprs==max(fprs[fprs < 0.05])])
    preds_binary = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds_binary).ravel()
    roc_auc = auc(fprs, tprs) 
    pr_precisions, pr_recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(pr_recalls, pr_precisions)
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy_score(y_true, preds_binary),
        "precision": precision_score(y_true, preds_binary, zero_division=0),
        "recall": recall_score(y_true, preds_binary, zero_division=0),
        "f1_score": f1_score(y_true, preds_binary, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "tnr": tn / (fp + tn) if (fp + tn) > 0 else 0.0,
    }
    return metrics
    
def metrics_fairness(x_test, y_test, predictions, sensitive_attribute, attribute_threshold, all_metrics=False, variable_name=None):
    fprs, _, thresholds = roc_curve(y_test, predictions)    
    threshold = np.min(thresholds[fprs==max(fprs[fprs < 0.05])])
    preds_binary = (predictions >= threshold).astype(int)
    aequitas_df = pd.DataFrame(
        {
            "attribute": (x_test[sensitive_attribute]>=attribute_threshold).map({True: "g2", False: "g1"}),
            "preds": preds_binary,
            "y": y_test.values if isinstance(y_test, pd.Series) else y_test
        }
    )
    g = Group()
    aequitas_df["score"] = aequitas_df["preds"]
    aequitas_df["label_value"] = aequitas_df["y"]
    aequitas_results = g.get_crosstabs(aequitas_df, attr_cols=["attribute"])[0]
    recall_g2 = aequitas_results[aequitas_results["attribute_value"] == "g2"][["tpr"]].values[0][0]
    recall_g1 = aequitas_results[aequitas_results["attribute_value"] == "g1"][["tpr"]].values[0][0]
    fpr_g2 = aequitas_results[aequitas_results["attribute_value"] == "g2"][["fpr"]].values[0][0]
    fpr_g1 = aequitas_results[aequitas_results["attribute_value"] == "g1"][["fpr"]].values[0][0]
    fnr_g2 = 1 - recall_g2
    fnr_g1 = 1 - recall_g1
    if fpr_g1 >= fpr_g2:
        fpr_ratio = fpr_g1 and fpr_g2/fpr_g1 or 0
    else:
        fpr_ratio = fpr_g2 and fpr_g1/fpr_g2 or 0
    if fnr_g1 > fnr_g2:
        fnr_ratio = fnr_g2 and fnr_g2/fnr_g1 or 0
    else:
        fnr_ratio = fnr_g2 and fnr_g1/fnr_g2 or 0
    if variable_name is not None:
        if all_metrics:
            return {f"fpr_ratio_{variable_name}": fpr_ratio, f"fnr_ratio_{variable_name}": fnr_ratio, f"recall_{variable_name}_g2": recall_g2, f"recall_{variable_name}_g1": recall_g1, f"fpr_{variable_name}_g2": fpr_g2, f"fpr_{variable_name}_g1": fpr_g1, f"fnr_{variable_name}_g2": fnr_g2, f"fnr_{variable_name}_g1": fnr_g1}
        return {f"fpr_ratio_{variable_name}": fpr_ratio, f"fnr_ratio_{variable_name}": fnr_ratio}
    if all_metrics:
        return {"fpr_ratio": fpr_ratio, "fnr_ratio": fnr_ratio, "recall_g2": recall_g2, "recall_g1": recall_g1, "fpr_g2": fpr_g2, "fpr_g1": fpr_g1, "fnr_g2": fnr_g2, "fnr_g1": fnr_g1}
    return {"fpr_ratio": fpr_ratio, "fnr_ratio": fnr_ratio}
