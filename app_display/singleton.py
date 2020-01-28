import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies as dd
import dash_table as dtable
from plotly import graph_objects as go
from os.path import dirname, join
import pandas as pd
from flask import send_from_directory
from .overview import halved_div
from .utils import get_scoring_dict, random_colors
from mnistk import Tester


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
    tester = Tester()

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

        data_dict = {}
        data_dict["stat_props"] = dict_from_file(join(mod_dir, "network.json"))
        data_dict["stat_rank"] = dict_from_file(join(mod_dir, "rankings.json"))
        data_dict["dyn_props"] = dict_from_file(join(run_dir, "properties.json"))
        data_dict["dyn_rank"] = dict_from_file(join(run_dir, "rankings.json"))
        data_dict["confusion-dict"], data_dict["splits-dict"] = get_scoring_dict(
            join(SPHandler.result_dir, "exam_stats.h5"),
            join(SPHandler.mod_dir, "runs", SPHandler.run, "predictions.h5"),
        )

        SPHandler.tester.load_network(
            SPHandler.mod_name,
            run_dir,
            list(int(x) for x in data_dict["dyn_props"]["test accuracy"].keys()),
        )

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
                        hoverData={"points": [dict(curveNumber=1, x=epoch)]},
                        config=dict(displayModeBar=False),
                    ),
                ),
                html.P(
                    "AUC is ridiculously high every time, because there are a large number of true negative predictions."
                ),
                halved_div(
                    dcc.Graph(id="split-chart", config=dict(displayModeBar=False)),
                    dcc.Graph(
                        id="accuracy-heatmap",
                        hoverData={"points": [dict(x=0, y=0)]},
                        config=dict(displayModeBar=False),
                    ),
                ),
                html.Div(
                    [
                        html.P(
                            "Here you can test the network with samples from the test dataset and view the predictions"
                        ),
                        html.Div(id="network-area"),
                    ],
                    id="testing-samples",
                ),
                html.P(json.dumps(data_dict["stat_props"])),
                html.Div(
                    [
                        html.P(id="ranking-info"),
                        html.P(
                            [
                                html.Span(
                                    [
                                        html.Img(
                                            id="single-struct",
                                            src="/{}/network.svg".format(
                                                SPHandler.mod_name
                                            ),
                                            height=800,
                                        )
                                    ],
                                    className="marginnote",
                                )
                            ],
                            id="net-structure",
                        ),
                    ],
                    id="ranking-div",
                    style=dict(width="75%"),
                ),
                html.Div(
                    [
                        html.Div(json.dumps(v), id=k.replace("_", "-"))
                        for k, v in data_dict.items()
                    ]
                    + [
                        html.Div(
                            json.dumps({"previous": str(epoch), "current": str(epoch)}),
                            id="epoch-hovers",
                        ),
                        html.Div(
                            json.dumps(
                                {
                                    "pie-splits": random_colors(
                                        s=0.9, v=0.4, num_colors=10
                                    )
                                }
                            ),
                            id="graph-colors",
                        ),
                    ],
                    id="data-dict",
                    style=dict(display="none"),
                ),
            ],
            id="single-content",
        )
        return layout

    @staticmethod
    def loss_function(dyn_props):
        dyn_props = json.loads(dyn_props)
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
    def bar_graphs(hoverData, dyn_props, epoch_hover):
        dyn_props = json.loads(dyn_props)
        epochs = list(dyn_props["test loss"].keys())
        pt = hoverData["points"][0]
        if pt["curveNumber"] != 1:
            return [dash.no_update, dash.no_update, dash.no_update]
        epoch_hover = json.loads(epoch_hover)
        epoch_hover["previous"] = epoch_hover["current"]
        epoch_hover["current"] = str(pt["x"])

        def get_bars(col, title):
            xs = list(range(10))
            average = lambda x: sum(x) / len(x)

            def get_diff():
                cur_avg = average(dyn_props[col][epoch_hover["current"]])
                diff = cur_avg - average(dyn_props[col][epoch_hover["previous"]])
                if diff > 0:
                    return "{:.5f}, \u25b4 {:.5f}".format(cur_avg, diff)
                elif diff < 0:
                    return "{:.5f}, \u25be {:.5f}".format(cur_avg, -diff)
                else:
                    return "{:.5f}".format(cur_avg)

            def opacity(x):
                if x == epoch_hover["previous"]:
                    return 0.9
                elif x == epoch_hover["current"]:
                    return 0.7
                else:
                    return 0

            def graph_symbol(a, b):
                if a > b:
                    return "triangle-up"
                elif a < b:
                    return "triangle-down"
                else:
                    return "square"

            ans1 = {
                "data": [
                    go.Scatter(
                        x=xs,
                        y=[float(x) for x in dyn_props[col][str(epoch)]],
                        mode="markers",
                        marker=dict(
                            size=12 if str(epoch) == epoch_hover["current"] else 8,
                            symbol=[
                                graph_symbol(
                                    dyn_props[col][str(epoch)][i],
                                    dyn_props[col][epoch_hover["previous"]][i],
                                )
                                for i in range(10)
                            ],
                        ),
                        opacity=opacity(epoch),
                        name="epoch {}".format(epoch),
                        hoverinfo="all" if epoch in epoch_hover.values() else "none",
                    )
                    for epoch in epochs
                ],
                "layout": dict(
                    hovermode="closest",
                    font={"family": "et-book", "size": 15},
                    xaxis=dict(
                        title="{} at Epoch {} (avg. {})".format(
                            title, epoch_hover["current"], get_diff()
                        ),
                        dtick=1,
                        zeroline=False,
                    ),
                    yaxis=dict(showline=False, zeroline=False),
                    height="225",
                    margin={"l": 40, "b": 40, "r": 10, "t": 10},
                ),
            }
            return ans1

        return [
            get_bars("test accuracy", "Accuracy"),
            get_bars("test AUC", "AUC"),
            json.dumps(epoch_hover),
        ]

    @staticmethod
    def accuracy_heatmap(hoverData, confusion_dict):
        pt = hoverData["points"][0]
        if pt["curveNumber"] != 1:
            return dash.no_update
        cf_data = json.loads(confusion_dict)[str(pt["x"])]

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

        anno_text = [get_annotation(i, i, cf_data["text"][i][i]) for i in range(10)] + [
            get_annotation(i, j, cf_data["text"][j][i])
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
                    opacity=0.8,
                    zmin=0,
                    zmax=cf_data["wmax"] + 5,
                    name="Wrong",
                    hovertemplate="%{z} samples<br>Prediction: %{x} <br>Truth: %{y}<br><extra></extra>",
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
                    opacity=0.8,
                    zmin=cf_data["cmin"] - 200,
                    zmax=cf_data["cmax"] + 200,
                    name="Correct",
                    hovertemplate="%{z} samples<br>Prediction: %{x} <br>Truth: %{y}<br><extra></extra>",
                ),
            ],
            "layout": dict(
                hovermode="closest",
                font={"family": "et-book", "size": 15},
                width="450",
                height="450",
                yaxis=dict(title="Truth", dtick=1, zeroline=False),
                xaxis=dict(title="Prediction", dtick=1, zeroline=False),
                margin={"l": 40, "b": 50, "r": 40, "t": 40},
                annotations=anno_text,
                title=dict(
                    color="#000000", text="Confusion Matrix at Epoch {}".format(pt["x"])
                ),
            ),
        }
        return ans

    @staticmethod
    def pie_splits(accuData, epoch_hover, splits_dict, colors):
        epoch = int(json.loads(epoch_hover)["current"])
        colors = json.loads(colors)["pie-splits"]
        pt = accuData["points"][0]
        sp_data = json.loads(splits_dict)[str(epoch)][pt["y"]][pt["x"]]
        to9 = list(range(10))
        if max(sp_data) == 0:
            return dash.no_update
        return {
            "data": [
                go.Pie(
                    labels=to9,
                    values=sp_data,
                    customdata=["%0.3f" % (x) for x in sp_data],
                    opacity=0.7,
                    marker=dict(colors=colors, line=dict(color="#000000", width=0)),
                    hovertemplate="predicting %{label} with score %{customdata}<extra></extra>",
                    texttemplate="%{label} <br> %{customdata}",
                    textposition="inside",
                    showlegend=False,
                    hole=0.5,
                )
            ],
            "layout": dict(
                hovermode="closest",
                font={"family": "et-book", "size": 15},
                width="600",
                height="400",
                margin={"l": 50, "b": 50, "r": 20, "t": 20},
                uniformtext=dict(minsize=8, mode="hide"),
                annotations=[
                    dict(
                        font=dict(color="#000000"),
                        x=0.5,
                        y=0.5,
                        text="Truth = {}".format(pt["y"]),
                        showarrow=False,
                    )
                ],
            ),
        }

    @staticmethod
    def ranking_info(hoverData, stat_props, stat_rank, dyn_props, dyn_rank):
        pt = hoverData["points"][0]
        stat_props = json.loads(stat_props)
        stat_rank = json.loads(stat_rank)
        dyn_props = json.loads(dyn_props)
        dyn_rank = json.loads(dyn_rank)
        if pt["curveNumber"] != 1:
            return dash.no_update
        layout = [
            html.P(
                [
                    "Ranking data for {}, run {}, epoch {}".format(
                        stat_props["name"], dyn_props["run"], pt["x"]
                    )
                ]
            )
        ]
        return layout

    @staticmethod
    def callbacks():
        SPHandler.app.callback(
            dd.Output(component_id="loss-graph", component_property="figure"),
            [dd.Input(component_id="dyn-props", component_property="children")],
        )(SPHandler.loss_function)
        SPHandler.app.callback(
            [
                dd.Output(component_id="accu-bars", component_property="figure"),
                dd.Output(component_id="auc-bars", component_property="figure"),
                dd.Output(component_id="epoch-hovers", component_property="children"),
            ],
            [dd.Input(component_id="loss-graph", component_property="hoverData")],
            [
                dd.State(component_id="dyn-props", component_property="children"),
                dd.State(component_id="epoch-hovers", component_property="children"),
            ],
        )(SPHandler.bar_graphs)

        SPHandler.app.callback(
            dd.Output(component_id="ranking-info", component_property="children"),
            [dd.Input(component_id="loss-graph", component_property="hoverData")],
            [
                dd.State(component_id="stat-props", component_property="children"),
                dd.State(component_id="stat-rank", component_property="children"),
                dd.State(component_id="dyn-props", component_property="children"),
                dd.State(component_id="dyn-rank", component_property="children"),
            ],
        )(SPHandler.ranking_info)

        SPHandler.app.callback(
            dd.Output(component_id="accuracy-heatmap", component_property="figure"),
            [dd.Input(component_id="loss-graph", component_property="hoverData")],
            [dd.State(component_id="confusion-dict", component_property="children")],
        )(SPHandler.accuracy_heatmap)

        SPHandler.app.callback(
            dd.Output(component_id="split-chart", component_property="figure"),
            [
                dd.Input(
                    component_id="accuracy-heatmap", component_property="hoverData"
                ),
                dd.Input(component_id="epoch-hovers", component_property="children"),
            ],
            [
                dd.State(component_id="splits-dict", component_property="children"),
                dd.State(component_id="graph-colors", component_property="children"),
            ],
        )(SPHandler.pie_splits)

        @SPHandler.app.server.route("/<path:im_path>.svg")
        def net_struct(im_path):
            return send_from_directory(
                join(SPHandler.result_dir, dirname(im_path)), "network.svg"
            )
