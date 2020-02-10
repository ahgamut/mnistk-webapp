import dash
from re import sub


class Constants(object):
    result_dir = "./results/"
    db_path = "./results/summary.db"
    ignore_columns = [
        "activation",
        "groupname",
        "formname",
        "loss",
        "rank",
        "rank_gp",
        "rank_form",
        "rank_snap",
    ]
    x_columns = ["time", "#params", "memory per pass", "#layers", "#ops"]
    y_columns = (
        ["accuracy", "AUC"]
        + ["accuracy_{}".format(i) for i in range(10)]
        + ["AUC_{}".format(i) for i in range(10)]
    )
    fixer = lambda x: sub("[^A-Za-z0-9 _]+", "", x)


external_css = ["https://cdnjs.cloudflare.com/ajax/libs/tufte-css/1.7.2/tufte.css"]
app = dash.Dash("app", external_stylesheets=external_css)
app.config.suppress_callback_exceptions = True
server = app.server
