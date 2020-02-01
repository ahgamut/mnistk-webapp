import h5py
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from mnistk import NDArrayEncoder, NDArrayDecoder
import json


def dict_from_file(path):
    ans = None
    with open(path, "r") as f:
        ans = json.load(f, cls=NDArrayDecoder)
    return ans


def get_scoring_dict(true_path, pred_path):
    confusion = {}
    splits = {}
    with h5py.File(true_path, "r") as f_true:
        truth = np.array(f_true.get("truth"))
        with h5py.File(pred_path, "r") as f_pred:
            for epoch in f_pred.keys():
                preds = np.array(f_pred.get("{}/preds".format(epoch)))
                scores = np.array(f_pred.get("{}/scores".format(epoch)))
                cf = confusion_matrix(truth, preds)
                confusion[int(epoch)] = get_confusion_data(cf)
                splits[int(epoch)] = get_split_data(truth, preds, scores)
    return confusion, splits


def get_split_data(truth, preds, scores):
    answer = np.zeros((10, 10, 10), np.float32)
    for i in range(10):
        for j in range(10):
            pd = scores[(truth == i) & (preds == j)]
            if len(pd) != 0:
                answer[i, j] = np.mean(np.exp(pd), axis=0)
    return answer


def get_confusion_data(cf):
    cf_text = cf.astype("|U5")
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


def random_colors(s, v, num_colors=10, h_start=None):
    # https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    if h_start is None:
        h_start = np.random.randint(0, 359)
    hs = np.linspace(
        start=h_start, stop=h_start + 360, num=num_colors + 1, dtype=np.float32
    )[:-1]
    hs %= 360
    C = v * s
    X = C * (1 - np.abs(((hs / 60) % 2) - 1))
    m = v - C
    rgb0 = []
    for i in range(len(hs)):
        h = hs[i]
        if h >= 0 and h < 60:
            val = (C, X[i], 0)
        elif h >= 60 and h < 120:
            val = (X[i], C, 0)
        elif h >= 120 and h < 180:
            val = (0, C, X[i])
        elif h >= 180 and h < 240:
            val = (0, X[i], C)
        elif h >= 240 and h < 300:
            val = (X[i], 0, C)
        elif h >= 300 and h < 360:
            val = (C, 0, X[i])
        rgb0.append(val)

    rgb0 = np.array(rgb0, dtype=np.float32)
    rgb = np.int32(np.rint((rgb0 + m) * 255))
    rgb_str = ["#%0.2X%0.2X%0.2X" % (x[0], x[1], x[2]) for x in rgb]
    return rgb_str
