import h5py
import numpy as np
import json
import pandas as pd
import dash_html_components as html
import os
import boto3
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from mnistk import Tester, NDArrayEncoder, NDArrayDecoder
from mnistk.collect import get_records

# Callback handling Utilities #


def maybe_get_from_s3(file_loader):
    def confirm_local(*args):
        client = boto3.client("s3")
        for path in args:
            if os.path.exists(path):
                continue
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                client.download_file(os.environ.get("S3_BUCKET_NAME"), path, path)

        result = file_loader(*args)
        return result

    return confirm_local


@maybe_get_from_s3
def get_scoring_dict(true_path, pred_path):
    confusion = {}
    splits = {}
    samples = {}
    df = pd.DataFrame(
        data=np.zeros((10000, 2), dtype=np.int32), columns=["truth", "preds"]
    )
    with h5py.File(true_path, "r") as f_true:
        df["truth"] = f_true.get("truth")
    with h5py.File(pred_path, "r") as f_pred:
        for epoch in f_pred.keys():
            df["preds"] = f_pred.get("{}/preds".format(epoch))
            scores = np.array(f_pred.get("{}/scores".format(epoch)))
            cf = confusion_matrix(df["truth"], df["preds"])
            confusion[int(epoch)] = get_confusion_data(cf)
            splits[int(epoch)] = get_split_data(df["truth"], df["preds"], scores)
            samples[int(epoch)] = pick_samples(df)
    return confusion, splits, samples


@maybe_get_from_s3
def dict_from_file(path):
    ans = None
    with open(path, "r") as f:
        ans = json.load(f, cls=NDArrayDecoder)
    return ans


@maybe_get_from_s3
def weights_from_file(path):
    return torch.load(path, map_location=torch.device("cpu"),)


@maybe_get_from_s3
def check_existence(path):
    if os.path.exists(path):
        return True
    else:
        return False


def get_grad_data(mod_name, run_dir, epoch, samples):
    weights = weights_from_file(os.path.join(run_dir, "network-{}.pth".format(epoch)))
    sample = np.random.choice(samples, size=1)[0]
    imgs, scores = Tester.get_sample_grads(mod_name, weights, sample)
    return imgs, scores


def pick_samples(df, size=3):
    ans = {}
    for (t, p), g in df.groupby(["truth", "preds"]):
        k = int(t) * 10 + int(p)
        if len(g) > size:
            ans[k] = np.int32(np.random.choice(g.index, size=size, replace=False))
        elif len(g) > 0:
            ans[k] = np.int32(g.index)
        else:
            ans[k] = []
    return ans


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


def dict_to_string(dval):
    return json.dumps(dval, cls=NDArrayEncoder)


def dict_from_string(sval):
    return json.loads(sval, cls=NDArrayDecoder)


def get_property_records(dyn_props, stat_props):
    props = {k: v for k, v in dyn_props.items()}
    props.update(stat_props)
    records_list = get_records(props)
    records_dict = {}
    for i, k in enumerate(sorted(dyn_props["test loss"].keys())):
        records_dict[k] = records_list[i]
    return records_dict


# Layout Utilities #


def lv(label, value, vlabel=None):
    vlab = label if vlabel is None else vlabel
    return {"label": label, "value": "{}|{}".format(vlab, value)}


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
