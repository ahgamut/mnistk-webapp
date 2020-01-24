import h5py
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc


def get_confusion_dict(true_path, pred_path):
    ans = {}
    with h5py.File(true_path, "r") as f_true:
        truth = np.array(f_true.get("truth"))
        with h5py.File(pred_path, "r") as f_pred:
            for epoch in f_pred.keys():
                preds = np.array(f_pred.get("{}/preds".format(epoch)))
                ans[int(epoch)] = get_confusion_data(truth, preds)
    return ans


def get_confusion_data(truth, preds):
    cf = confusion_matrix(truth, preds)
    cf_text = cf.astype("|U10")

    cf_correct = np.diag(np.diag(cf.astype(np.float32)))
    cf_wrong = cf - cf_correct
    a = cf_correct == 0
    cf_correct[a] = np.nan
    cf_wrong[a == 0] = np.nan
    ans = dict(
        wrong=cf_wrong.tolist(),
        wmin=np.nanmin(cf_wrong).tolist(),
        wmax=np.nanmax(cf_wrong).tolist(),
        correct=cf_correct.tolist(),
        cmin=np.nanmin(cf_correct).tolist(),
        cmax=np.nanmax(cf_correct).tolist(),
        text=cf_text.tolist(),
    )
    return ans
