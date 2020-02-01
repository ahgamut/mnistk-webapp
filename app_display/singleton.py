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
from .utils import get_scoring_dict, random_colors, dict_from_file
from mnistk import Tester, NDArrayEncoder, NDArrayDecoder
from mnistk.collect import get_records


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


def set_layout(pathname0):
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
    data_dict["stat-props"] = dict_from_file(join(mod_dir, "network.json"))
    data_dict["stat-rank"] = dict_from_file(join(mod_dir, "rankings.json"))
    data_dict["dyn-props"] = dict_from_file(join(run_dir, "properties.json"))
    data_dict["dyn-rank"] = dict_from_file(join(run_dir, "rankings.json"))
    data_dict["confusion-dict"], data_dict["splits-dict"] = get_scoring_dict(
        join(SPHandler.result_dir, "exam_stats.h5"),
        join(SPHandler.mod_dir, "runs", SPHandler.run, "predictions.h5"),
    )
    data_dict["dyn-records"] = {}
    dp = {k: v for k, v in data_dict["dyn-props"].items()}
    dp.update(data_dict["stat-props"])
    dyn_records = get_records(dp)
    for i, k in enumerate(sorted(data_dict["dyn-props"]["test loss"].keys())):
        data_dict["dyn-records"][k] = dyn_records[i]

    SPHandler.tester.load_network(
        SPHandler.mod_name,
        run_dir,
        list(int(x) for x in data_dict["dyn-props"]["test accuracy"].keys()),
    )
    layout = html.Div(
        children=[
            html.H2("Performance Details"),
            halved_div(
                html.Div(
                    [
                        dcc.Graph(id="accu-bars", config=dict(displayModeBar=False)),
                        dcc.Graph(id="auc-bars", config=dict(displayModeBar=False)),
                    ]
                ),
                dcc.Graph(
                    id="loss-graph",
                    hoverData={"points": [dict(curveNumber=1, x=epoch)]},
                    clickData={"points": [dict(curveNumber=1, x=epoch)]},
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
                    hoverData={
                        "points": [
                            dict(
                                x=0,
                                y=0,
                                z=data_dict["confusion-dict"][epoch]["correct"][0][0],
                            )
                        ]
                    },
                    config=dict(displayModeBar=False),
                ),
            ),
            html.H2("View gradients with individual samples"),
            html.Div(
                [
                    html.P(
                        "Here you can test the network with samples from the test dataset and view the predictions"
                    ),
                    html.Div(id="network-area"),
                ],
                id="testing-samples",
            ),
            html.H2("Structure and Rankings"),
            html.Div(
                [
                    html.P(
                        [
                            html.Span(
                                [
                                    html.Img(
                                        id="single-struct",
                                        src="/{}/network.svg".format(
                                            SPHandler.mod_name
                                        ),
                                        style={"max-height": 1100, "max-width": 400,},
                                    )
                                ],
                                className="marginnote",
                            )
                        ],
                        id="net-structure",
                    ),
                    html.P(id="ranking-info"),
                ],
                id="ranking-div",
                style=dict(width="75%"),
            ),
            html.Div(
                [
                    html.Div(json.dumps(v, cls=NDArrayEncoder), id=k)
                    for k, v in data_dict.items()
                ]
                + [
                    html.Div(str(epoch), id="current-epoch"),
                    html.Div(
                        json.dumps(
                            {"pie-splits": random_colors(s=0.85, v=0.4, num_colors=10)}
                        ),
                        id="graph-colors",
                    ),
                ],
                id="data-dict",
                style=dict(display="none"),
            ),
            dcc.Link("Go Back to Overview", href="/"),
        ],
        id="single-content",
    )
    return layout


def loss_function(lossHover, lossClick, dyn_props, current_epoch):
    pt_hover = lossHover["points"][0]
    pt_click = lossClick["points"][0]
    if pt_hover["curveNumber"] != 1 or pt_click["curveNumber"] != 1:
        return dash.no_update, dash.no_update
    dyn_props = json.loads(dyn_props, cls=NDArrayDecoder)
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
                marker=dict(
                    size=12,
                    line=dict(
                        width=[1.5 * (t[0] == pt_click["x"]) for t in testdata],
                        color="#000000",
                    ),
                ),
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

    epoch = str(pt_click["x"])
    if epoch == current_epoch:
        epoch = dash.no_update
    return ans, epoch


def bar_graphs(lossHover, current_epoch, dyn_props):
    pt_hover = lossHover["points"][0]
    if pt_hover["curveNumber"] != 1:
        return dash.no_update, dash.no_update
    dyn_props = json.loads(dyn_props, cls=NDArrayDecoder)
    epochs = list(dyn_props["test loss"].keys())
    highlight_epoch = str(pt_hover["x"])

    def get_bars(col, title):
        xs = list(range(10))
        average = lambda x: sum(x) / len(x)

        def opacity(x):
            if x == highlight_epoch:
                return 0.7
            elif x == current_epoch:
                return 0.9
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
                    y=[float(x) for x in dyn_props[col][epoch]],
                    mode="markers",
                    marker=dict(
                        size=12 if epoch == current_epoch else 8,
                        symbol=[
                            graph_symbol(
                                dyn_props[col][epoch][i],
                                dyn_props[col][current_epoch][i],
                            )
                            for i in range(10)
                        ],
                    ),
                    opacity=opacity(epoch),
                    name="epoch {}".format(epoch),
                    hoverinfo="all"
                    if epoch in [highlight_epoch, current_epoch]
                    else "none",
                )
                for epoch in epochs
            ],
            "layout": dict(
                hovermode="closest",
                font={"family": "et-book", "size": 15},
                xaxis=dict(
                    title="{} at Epoch {} (avg. {})".format(
                        title, current_epoch, average(dyn_props[col][current_epoch])
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

    return [get_bars("test accuracy", "Accuracy"), get_bars("test AUC", "AUC")]


def ranking_info(current_epoch, stat_props, stat_rank, dyn_records, dyn_rank):
    epoch = current_epoch
    stat_props = json.loads(stat_props, cls=NDArrayDecoder)
    stat_rank = json.loads(stat_rank, cls=NDArrayDecoder)
    dyn_record = json.loads(dyn_records, cls=NDArrayDecoder)[epoch]
    dyn_rank = json.loads(dyn_rank, cls=NDArrayDecoder)[epoch]

    def rankdf(df_dict, score_dict):
        df = pd.DataFrame(columns=df_dict["columns"], data=df_dict["data"])
        df["metric"] = df_dict["index"]
        df["value"] = [score_dict.get(x, -1) for x in df["metric"]]
        df = df[["metric", "value"] + df_dict["columns"]]
        out_of = df[df["metric"] == "out of"]
        df = df[df["metric"] != "out of"]
        return df, out_of

    stat_df, stat_out = rankdf(stat_rank, stat_props)
    dyn_df, dyn_out = rankdf(dyn_rank, dyn_record)

    header_names = {
        "Metric": "metric",
        "Value": "value",
        "Global Rank": "global",
        "Group Rank": "in_group",
        "Form Rank": "in_form",
        "Run Rank": "in_run",
    }
    table_props = dict(
        columns=[{"name": k, "id": v} for k, v in header_names.items()],
        style_table={"overflowX": "scroll"},
        style_cell={
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "font-family": "et-book",
            "font-size": 18,
        },
        style_data_conditional=[
            {
                "if": dict(column_id=x, filter_query="{%s} eq 1" % (x)),
                "fontWeight": "bold",
                "backgroundColor": "rgb(102,255,51)",
            }
            for x in ["global", "in_group", "in_form", "in_run"]
        ],
        style_header={
            "backgroundColor": "rgb(230, 230, 230)",
            "fontWeight": "bold",
            "font-family": "et-book",
            "font-size": 20,
        },
    )
    stat_table = dtable.DataTable(
        id="stat-table", data=stat_df.to_dict("records"), **table_props
    )
    dyn_table = dtable.DataTable(
        id="dyn-table", data=dyn_df.to_dict("records"), **table_props
    )
    layout = [
        html.Div(
            [
                html.P("Ranking Structural data for {}".format(stat_props["name"])),
                stat_table,
            ]
        ),
        html.Br(),
        html.Div(
            [
                html.P(
                    "Ranking Prediction data for {}, run {}, tested on epoch {}:".format(
                        stat_props["name"], dyn_record["run"], current_epoch
                    )
                ),
                dyn_table,
            ]
        ),
    ]
    return layout


def accuracy_heatmap(current_epoch, confusion_dict, accuHover):
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

    cf_data = json.loads(confusion_dict, cls=NDArrayDecoder)[current_epoch]
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
                hovertemplate="#Samples: %{z}<br>Truth: %{y}<br>Prediction: %{x} <br><extra></extra>",
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
                hovertemplate="#Samples: %{z}<br>Truth: %{y}<br>Prediction: %{x} <br><extra></extra>",
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
                color="#000000",
                text="Confusion Matrix at Epoch {}".format(current_epoch),
            ),
        ),
    }
    acc_pt = accuHover["points"][0]
    acc_pt["z"] = int(cf_data["text"][acc_pt["y"]][acc_pt["x"]])
    if acc_pt["z"] == 0:
        acc_pt["x"] = 0
        acc_pt["y"] = 0
        acc_pt["z"] = int(cf_data["text"][acc_pt["y"]][acc_pt["x"]])
    return ans, accuHover


def pie_splits(accuHover, current_epoch, splits_dict, colors):
    epoch = int(current_epoch)
    pt = accuHover["points"][0]
    sp_data = json.loads(splits_dict, cls=NDArrayDecoder)[str(epoch)][pt["y"], pt["x"]]
    to9 = list(range(10))
    if max(sp_data) == 0:
        return dash.no_update
    colors = json.loads(colors)["pie-splits"]
    return {
        "data": [
            go.Pie(
                labels=to9,
                values=sp_data,
                customdata=["<b>%0.3f</b>" % (x) for x in sp_data],
                opacity=0.9,
                marker=dict(colors=colors, line=dict(color="#000000", width=0.5)),
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
            width="450",
            height="450",
            margin={"l": 20, "b": 35, "r": 20, "t": 40},
            uniformtext=dict(minsize=8, mode="hide"),
            annotations=[
                dict(
                    font=dict(color="#000000", size=18),
                    x=0.5,
                    y=0.5,
                    text="{} Sample{}<br>Truth: {}<br>Prediction: {}".format(
                        pt["z"], "s" if pt["z"] > 1 else "", pt["y"], pt["x"]
                    ),
                    showarrow=False,
                )
            ],
            title=dict(text="Distribution of Prediction Scores"),
        ),
    }


def set_callbacks(result_dir, app):
    sh = SPHandler(result_dir, app)
    app.callback(
        [dd.Output("loss-graph", "figure"), dd.Output("current-epoch", "children")],
        [dd.Input("loss-graph", "hoverData"), dd.Input("loss-graph", "clickData")],
        [dd.State("dyn-props", "children"), dd.State("current-epoch", "children")],
    )(loss_function)
    app.callback(
        [dd.Output("accu-bars", "figure"), dd.Output("auc-bars", "figure")],
        [dd.Input("loss-graph", "hoverData"), dd.Input("current-epoch", "children"),],
        [dd.State("dyn-props", "children")],
    )(bar_graphs)

    app.callback(
        dd.Output("ranking-info", "children"),
        [dd.Input("current-epoch", "children")],
        [
            dd.State("stat-props", "children"),
            dd.State("stat-rank", "children"),
            dd.State("dyn-records", "children"),
            dd.State("dyn-rank", "children"),
        ],
    )(ranking_info)

    app.callback(
        [
            dd.Output("accuracy-heatmap", "figure"),
            dd.Output("accuracy-heatmap", "hoverData"),
        ],
        [dd.Input("current-epoch", "children")],
        [
            dd.State("confusion-dict", "children"),
            dd.State("accuracy-heatmap", "hoverData"),
        ],
    )(accuracy_heatmap)

    app.callback(
        dd.Output("split-chart", "figure"),
        [dd.Input("accuracy-heatmap", "hoverData")],
        [
            dd.State("current-epoch", "children"),
            dd.State("splits-dict", "children"),
            dd.State("graph-colors", "children"),
        ],
    )(pie_splits)

    @app.server.route("/<path:im_path>.svg")
    def net_struct(im_path):
        return send_from_directory(join(result_dir, dirname(im_path)), "network.svg")
