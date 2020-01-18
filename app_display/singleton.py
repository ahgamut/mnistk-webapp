import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies as dd
import dash_table as dtable
from plotly import graph_objects as go, figure_factory as ff
from os.path import dirname, join
import pandas as pd
from flask import send_from_directory
from .overview import halved_div
from .utils import get_confusion_split


def dict_from_file(path):
    ans = None
    with open(path, "r") as f:
        ans = json.load(f)
    return ans


class SPHandler(object):
    result_dir = None
    app = None
    mod_dir = None
    mod_name = None
    run = None
    epoch = None
    stat_props = None
    stat_rank = None
    dyn_props = None
    dyn_rank = None

    def __init__(self, result_dir, app):
        SPHandler.result_dir = result_dir
        SPHandler.app = app
        SPHandler.callbacks()

    @staticmethod
    def layout(pathname0):
        pathname = pathname0.split("/")
        SPHandler.mod_name = pathname[1]
        SPHandler.mod_dir = join(SPHandler.result_dir, pathname[1])
        SPHandler.run = pathname[2]
        SPHandler.epoch = int(pathname[3])

        mod_dir = SPHandler.mod_dir
        run = SPHandler.run
        epoch = SPHandler.epoch
        run_dir = join(mod_dir, "runs", run)

        SPHandler.stat_props = dict_from_file(join(mod_dir, "network.json"))
        SPHandler.stat_rank = dict_from_file(join(mod_dir, "rankings.json"))
        SPHandler.dyn_props = dict_from_file(join(run_dir, "properties.json"))
        SPHandler.dyn_rank = dict_from_file(join(run_dir, "rankings.json"))

        layout = html.Div(
            children=[
                dcc.Link("Go Back to Overview", href="/"),
                halved_div(
                    html.Div(
                        [
                            dcc.Graph(
                                id="accu-bars", config=dict(displayModeBar=False)
                            ),
                            dcc.Graph(id="auc-bars", config=dict(displayModeBar=False)),
                        ]
                    ),
                    dcc.Graph(
                        id="loss-graph",
                        figure=SPHandler.loss_function(),
                        clickData={"points": [dict(curveNumber=1, x=epoch)]},
                        config=dict(displayModeBar=False),
                    ),
                ),
                html.P(
                    "AUC is ridiculously high every time, because there are a large number of true negative predictions."
                ),
                dcc.Graph(id="accuracy-heatmap", config=dict(displayModeBar=False)),
                html.P(json.dumps(SPHandler.stat_props)),
                html.Div(id="ranking-info", style=dict(width="75%")),
            ],
            id="single-content",
        )
        return layout

    @staticmethod
    def loss_function():
        dyn_props = SPHandler.dyn_props
        testdata = sorted(
            ((int(x[0]), float(x[1])) for x in dyn_props["test loss"].items()),
            key=lambda x: int(x[0]),
        )
        ans = {
            "data": [
                go.Scatter(
                    x=list(range(1, len(dyn_props["train loss"]) + 1)),
                    y=dyn_props["train loss"],
                    mode="lines+markers",
                    marker=dict(size=12),
                    opacity=0.6,
                    name="train",
                ),
                go.Scatter(
                    x=[x[0] for x in testdata],
                    y=[x[1] for x in testdata],
                    mode="markers",
                    marker=dict(size=12),
                    opacity=0.6,
                    name="test",
                ),
            ],
            "layout": dict(
                hovermode="closest",
                xaxis=dict(title="Epoch"),
                yaxis=dict(title="Loss"),
                font={"family": "et-book", "size": 15},
                clickmode="event",
                height="450",
                margin={"l": 50, "b": 40, "r": 10, "t": 10},
            ),
        }
        return ans

    @staticmethod
    def bar_graphs(clickData):
        pt = clickData["points"][0]
        if pt["curveNumber"] != 1:
            return [dash.no_update, dash.no_update]
        return [
            SPHandler.get_bars(pt["x"], "test accuracy", "Accuracy"),
            SPHandler.get_bars(pt["x"], "test AUC", "AUC"),
        ]

    @staticmethod
    def get_bars(epoch, col, title):
        dyn_props = SPHandler.dyn_props
        xs = list(range(10))
        ys = [float(x) for x in dyn_props[col][str(epoch)]]
        ans1 = {
            "data": [go.Bar(x=xs, y=ys, width=0.4, opacity=0.6)],
            "layout": dict(
                hovermode="closest",
                font={"family": "et-book", "size": 15},
                xaxis=dict(title="{} at epoch {}".format(title, epoch), dtick=1),
                yaxis={"range": [min(ys) - 0.01, 1]},
                height="225",
                margin={"l": 40, "b": 40, "r": 10, "t": 10},
            ),
        }
        return ans1

    @staticmethod
    def ranking_info(clickData):
        pt = clickData["points"][0]
        if pt["curveNumber"] != 1:
            return dash.no_update
        layout = [
            html.P(
                [
                    "Ranking data for {}, run {}, epoch {}".format(
                        SPHandler.mod_name, SPHandler.run, pt["x"]
                    ),
                    html.Span(
                        className="marginnote",
                        children=[
                            html.Img(
                                id="single-struct",
                                src="/{}/network.svg".format(SPHandler.mod_name),
                                height=800,
                            )
                        ],
                    ),
                ]
            )
        ]
        return layout

    @staticmethod
    def accuracy_heatmap(clickData):
        pt = clickData["points"][0]
        if pt["curveNumber"] != 1:
            return dash.no_update
        cf_data = get_confusion_split(
            join(SPHandler.result_dir, "exam_stats.h5"),
            join(SPHandler.mod_dir, "runs", SPHandler.run, "predictions.h5"),
            pt["x"],
        )

        def get_annotation(x, y, text):
            return dict(
                font=dict(color="#000000"),
                showarrow=False,
                text=text,
                x=x,
                y=y,
                xref="x",
                yref="y",
            )

        anno_text = [get_annotation(i, i, cf_data["text"][i, i]) for i in range(10)] + [
            get_annotation(i, j, cf_data["text"][j, i])
            for i in range(10)
            for j in range(10)
            if i != j
        ]
        to9 = list(range(10))
        ans = {
            "data": [
                go.Heatmap(
                    x=to9,
                    xgap=3,
                    y=to9,
                    ygap=3,
                    z=cf_data["wrong"],
                    showscale=False,
                    hoverongaps=False,
                    colorscale="Reds",
                    zmin=0,
                    zmax=cf_data["wmax"] + 5,
                ),
                go.Heatmap(
                    x=to9,
                    xgap=3,
                    y=to9,
                    ygap=3,
                    z=cf_data["correct"],
                    showscale=False,
                    hoverongaps=False,
                    colorscale="Greens",
                    zmin=cf_data["cmin"] - 200,
                    zmax=cf_data["cmax"] + 200,
                ),
            ],
            "layout": dict(
                hovermode="closest",
                font={"family": "et-book", "size": 15},
                width="400",
                height="400",
                xaxis=dict(title="Truth", dtick=1, zeroline=False),
                yaxis=dict(title="Prediction", dtick=1, zeroline=False),
                margin={"l": 40, "b": 50, "r": 0, "t": 0},
                annotations=anno_text,
            ),
        }
        return ans

    @staticmethod
    def callbacks():
        SPHandler.app.callback(
            [
                dd.Output(component_id="accu-bars", component_property="figure"),
                dd.Output(component_id="auc-bars", component_property="figure"),
            ],
            [dd.Input(component_id="loss-graph", component_property="clickData")],
        )(SPHandler.bar_graphs)

        SPHandler.app.callback(
            dd.Output(component_id="ranking-info", component_property="children"),
            [dd.Input(component_id="loss-graph", component_property="clickData")],
        )(SPHandler.ranking_info)

        SPHandler.app.callback(
            dd.Output(component_id="accuracy-heatmap", component_property="figure"),
            [dd.Input(component_id="loss-graph", component_property="clickData")],
        )(SPHandler.accuracy_heatmap)

        @SPHandler.app.server.route("/<path:im_path>.svg")
        def net_struct(im_path):
            return send_from_directory(SPHandler.mod_dir, "network.svg")
