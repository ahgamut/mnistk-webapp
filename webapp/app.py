# -*- coding: utf-8 -*-
import dash
import re
import os


class Constants(object):
    using_local_data = os.environ.get("S3_BUCKET_NAME", None) is None
    result_dir = "results/"
    db_path = "results/summary.db"
    ignore_columns = [
        "activation",
        "groupname",
        "formname",
        "loss",
        "rank",
        "rank_gp",
        "rank_form",
        "rank_snap",
        "#layers",
    ]
    x_columns = ["time", "#params", "memory per pass", "#ops"]
    y_columns = (
        ["accuracy", "AUC"]
        + ["accuracy_{}".format(i) for i in range(10)]
        + ["AUC_{}".format(i) for i in range(10)]
    )
    pretty_names = {
        "time": "Time",
        "#layers": "Layers",
        "#params": "Parameters",
        "memory per pass": "Memory Usage",
        "#ops": "Operations",
        "name": "Name",
        "run": "Run",
        "epoch": "Epoch",
        "accuracy": "Accuracy",
    }
    regex_fix = re.compile("[^\w ]+")


external_css = ["https://cdnjs.cloudflare.com/ajax/libs/tufte-css/1.7.2/tufte.css"]
app = dash.Dash("app", external_stylesheets=external_css)
app.config.suppress_callback_exceptions = True
server = app.server
