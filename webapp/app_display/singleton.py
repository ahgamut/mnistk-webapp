# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies as dd
import dash_table as dtable
from plotly import graph_objects as go
from pandas import DataFrame
from flask import send_from_directory
from os.path import dirname, join
from shutil import rmtree
from .utils import (
    get_scoring_dict,
    get_grad_data,
    get_property_records,
    check_existence,
    dict_from_file,
    dict_from_string,
    dict_to_string,
    random_colors,
)
from .overview import halved_div
from .textinfo import (
    ranking_text_0,
    prediction_text_0,
    prediction_text_1,
    prediction_text_2,
    heatmap_text_0,
    loss_text_0,
)
from app import app, Constants

####################
# Layout functions #
####################


def load_network_info(pathname0):
    pathname = pathname0.split("/")
    mod_name = pathname[1]
    mod_dir = join(Constants.result_dir, pathname[1])
    run = pathname[2]
    epoch = int(pathname[3])
    run_dir = join(mod_dir, "runs", run)

    info_dict = {"current-mod": mod_name, "current-run": run, "current-epoch": epoch}
    info_dict["stat-props"] = dict_from_file(join(mod_dir, "network.json"))
    info_dict["stat-rank"] = dict_from_file(join(mod_dir, "rankings.json"))
    info_dict["dyn-props"] = dict_from_file(join(run_dir, "properties.json"))
    info_dict["dyn-rank"] = dict_from_file(join(run_dir, "rankings.json"))
    (
        info_dict["confusion-dict"],
        info_dict["splits-dict"],
        info_dict["samples-dict"],
    ) = get_scoring_dict(
        join(Constants.result_dir, "exam_stats.h5"),
        join(mod_dir, "runs", run, "predictions.h5"),
    )
    info_dict["dyn-records"] = get_property_records(
        info_dict["dyn-props"], info_dict["stat-props"]
    )
    return info_dict


def layout_loss(epoch):
    div = halved_div(
        html.Div(
            [
                dcc.Graph(id="accu-bars", config=dict(displayModeBar=False)),
                dcc.Graph(id="auc-bars", config=dict(displayModeBar=False)),
            ]
        ),
        dcc.Graph(
            id="loss-graph",
            hoverData={"points": [dict(curveNumber=1, x=epoch)]},
            config=dict(displayModeBar=False),
        ),
        50,
    )
    return [
        html.H2("Performance Across Epochs"),
        dcc.Markdown(loss_text_0, style=dict(width="75%")),
        div,
    ]


def layout_detail(zval):
    div = halved_div(
        dcc.Graph(id="split-chart", config=dict(displayModeBar=False)),
        dcc.Graph(
            id="accuracy-heatmap",
            hoverData={"points": [dict(x=0, y=0, z=zval,)]},
            config=dict(displayModeBar=False),
        ),
    )
    return [
        html.H2("Distribution of Predictions"),
        dcc.Markdown(heatmap_text_0, style=dict(width="75%")),
        div,
    ]


def layout_testing():
    to9 = [{"label": str(x), "value": str(x)} for x in range(10)]
    t = dcc.Dropdown(id="testing-truth", options=to9, value="0", clearable=False)
    p = dcc.Dropdown(id="testing-preds", options=to9, value="0", clearable=False)
    button = html.Button(
        id="grads-button",
        children="Get Gradients",
        style={"font-family": "et-book", "font-size": 20,},
    )
    f = lambda x, y: html.Div(
        x, style=dict(display="table-cell", width="{}%".format(y))
    )
    opt_div = html.Div(
        [
            html.Div(
                [
                    f(html.P("Ground Truth"), 25),
                    f("", 5),
                    f(t, 5),
                    f("", 5),
                    f(html.P("Network Prediction"), 25),
                    f("", 5),
                    f(p, 5),
                    f("", 25),
                ],
                style=dict(display="table-row"),
            ),
            html.Div(button, style=dict(display="table-row")),
        ],
        style=dict(display="table"),
    )
    radio_div = halved_div(
        dcc.RadioItems(
            id="grads-colorbar",
            options=[
                {"label": "Across all gradients  ", "value": "global"},
                {"label": "Across this gradient  ", "value": "local"},
                {"label": "Hide the gradient map ", "value": "none"},
            ],
            value="global",
            style={"font-size": 20},
        ),
        html.P("Set gradient map colorbar range: "),
        32,
    )
    figs_div = halved_div(
        dcc.Graph(id="grad-images", config=dict(displayModeBar=False),),
        dcc.Graph(id="grad-scores", config=dict(displayModeBar=False)),
        32,
    )
    return [
        html.H2("How a prediction occurred"),
        html.Div(
            [
                dcc.Markdown(prediction_text_0, style=dict(width="75%")),
                opt_div,
                html.Div(
                    [
                        dcc.Markdown(prediction_text_1, style=dict(width="75%")),
                        radio_div,
                        figs_div,
                        html.Div("0", id="current-pred", style=dict(display="none")),
                        html.Div(id="gdd-scores", style=dict(display="none")),
                        html.Div(id="gdd-images", style=dict(display="none")),
                        dcc.Markdown(prediction_text_2, style=dict(width="75%")),
                    ],
                    id="network-area",
                    style=dict(display="none"),
                ),
                html.Div("", id="network-error", style=dict(color="red")),
            ],
            id="testing-samples",
        ),
    ]


def layout_rankings(mod_name):
    return [
        html.H2("Structure and Rankings"),
        html.Div(
            [
                html.P(
                    [
                        html.Span(
                            [
                                html.Img(
                                    id="single-struct",
                                    src="/{}/network.svg".format(mod_name),
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
    ]


def layout_hidden(info_dict, epoch):
    return [
        html.Div(
            [html.Div(dict_to_string(v), id=k) for k, v in info_dict.items()]
            + [
                html.Div(
                    dict_to_string(
                        {"pie-splits": random_colors(s=0.85, v=0.4, num_colors=10)}
                    ),
                    id="graph-colors",
                ),
            ],
            id="data-dict",
            style=dict(display="none"),
        ),
    ]


def set_layout(pathname0):
    info_dict = load_network_info(pathname0)
    mod_name = info_dict["current-mod"]
    run = info_dict["current-run"]
    epoch = info_dict["current-epoch"]
    subtitle = html.P(
        children="{}, run {}, epoch {}".format(mod_name, run, epoch),
        className="subtitle",
    )
    f = lambda x, y: html.Div(
        x, style=dict(display="table-cell", width="{}%".format(y))
    )
    nav_list = (
        [f(subtitle, 65),]
        + [
            f(
                dcc.Link(
                    "Epoch {}".format(x), href="/{}/{}/{}".format(mod_name, run, x)
                ),
                5,
            )
            for x in info_dict["dyn-props"]["test loss"].keys()
            if str(x) != str(epoch)
        ]
        + [f(dcc.Link("Go Back to Overview", href="/"), 15)]
    )
    nav_list2 = [f("", 65)] + nav_list[1:]
    navigation = html.Div(
        children=nav_list,
        style=dict(display="table", align="right", cellspacing="3px"),
    )
    navigation2 = html.Div(
        children=nav_list2,
        style=dict(display="table", align="left", cellspacing="3px"),
    )

    layout = html.Div(
        children=[navigation]
        + layout_loss(epoch)
        + layout_detail(info_dict["confusion-dict"][epoch]["correct"][0][0])
        + layout_testing()
        + layout_rankings(info_dict["current-mod"])
        + layout_hidden(info_dict, epoch)
        + [navigation2],
        id="single-content",
    )
    return layout


######################
# Callback functions #
######################


@app.callback(
    dd.Output("loss-graph", "figure"),
    [dd.Input("loss-graph", "hoverData"), dd.Input("current-epoch", "children")],
    [dd.State("dyn-props", "children")],
)
def loss_function(lossData, current_epoch, dyn_props):
    pt_hover = lossData["points"][0]
    if pt_hover["curveNumber"] != 1:
        return dash.no_update
    dyn_props = dict_from_string(dyn_props)
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
                        width=[
                            1.5 * (t[0] == pt_hover["x"])
                            + 2.5 * (t[0] == int(current_epoch))
                            for t in testdata
                        ],
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
            width="500",
            height="450",
            margin={"l": 50, "b": 40, "r": 10, "t": 10},
        ),
    }
    return ans


@app.callback(
    [dd.Output("accu-bars", "figure"), dd.Output("auc-bars", "figure")],
    [dd.Input("loss-graph", "hoverData"), dd.Input("current-epoch", "children"),],
    [dd.State("dyn-props", "children")],
)
def bar_graphs(lossData, current_epoch, dyn_props):
    pt_hover = lossData["points"][0]
    if pt_hover["curveNumber"] != 1:
        return dash.no_update, dash.no_update
    dyn_props = dict_from_string(dyn_props)
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
                        size=12 if epoch == current_epoch else 11,
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
                    title="{} at Epoch {} (avg. {:.6f})".format(
                        title, current_epoch, average(dyn_props[col][current_epoch])
                    ),
                    dtick=1,
                    zeroline=False,
                ),
                yaxis=dict(showline=False, zeroline=False),
                width="500",
                height="225",
                margin={"l": 40, "b": 40, "r": 10, "t": 10},
            ),
        }
        return ans1

    return [get_bars("test accuracy", "Accuracy"), get_bars("test AUC", "AUC")]


@app.callback(
    dd.Output("ranking-info", "children"),
    [dd.Input("current-epoch", "children")],
    [
        dd.State("stat-props", "children"),
        dd.State("stat-rank", "children"),
        dd.State("dyn-records", "children"),
        dd.State("dyn-rank", "children"),
    ],
)
def ranking_info(current_epoch, stat_props, stat_rank, dyn_records, dyn_rank):
    epoch = current_epoch
    stat_props = dict_from_string(stat_props)
    stat_rank = dict_from_string(stat_rank)
    dyn_record = dict_from_string(dyn_records)[epoch]
    dyn_rank = dict_from_string(dyn_rank)[epoch]

    def rankdf(df_dict, score_dict):
        df = DataFrame(columns=df_dict["columns"], data=df_dict["data"])
        df["metric"] = df_dict["index"]
        df["value"] = [score_dict.get(x, 1) for x in df["metric"]]
        df = df[["metric", "value"] + df_dict["columns"]]
        out_of = dict(df[df["metric"] == "out of"].iloc[0])
        df = df[df["metric"] != "out of"]
        df["metric"] = df["metric"].map(lambda x: Constants.pretty_names.get(x, x))
        df.round(10)
        return df, out_of

    stat_df, stat_out = rankdf(stat_rank, stat_props)
    dyn_df, dyn_out = rankdf(dyn_rank, dyn_record)
    dyn_df["value"] = dyn_df["value"].round(10)
    header_names = lambda outval: {
        "Metric": "metric",
        "Value": "value",
        "Global (/{})".format(outval["global"]): "global",
        "Group (/{})".format(outval["in_group"]): "in_group",
        "Form (/{})".format(outval["in_form"]): "in_form",
        "Run (/{})".format(outval["in_run"]): "in_run",
    }
    table_props = lambda outval: dict(
        columns=[{"name": k, "id": v} for k, v in header_names(outval).items()],
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
        id="stat-table", data=stat_df.to_dict("records"), **table_props(stat_out)
    )
    dyn_table = dtable.DataTable(
        id="dyn-table", data=dyn_df.to_dict("records"), **table_props(dyn_out)
    )
    text_infodict = {
        x: dyn_record[x] for x in ["groupname", "formname", "run", "epoch"]
    }
    layout = [
        dcc.Markdown(ranking_text_0.format(**text_infodict)),
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
                    "Ranking Run-Dependent data for {}, run {}, tested on epoch {}:".format(
                        stat_props["name"], dyn_record["run"], current_epoch
                    )
                ),
                dyn_table,
            ]
        ),
    ]
    return layout


@app.callback(
    [
        dd.Output("accuracy-heatmap", "figure"),
        dd.Output("accuracy-heatmap", "hoverData"),
    ],
    [dd.Input("current-epoch", "children")],
    [
        dd.State("confusion-dict", "children"),
        dd.State("accuracy-heatmap", "hoverData"),
    ],
)
def accuracy_heatmap(current_epoch, confusion_dict, accuData):
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

    cf_data = dict_from_string(confusion_dict)[current_epoch]
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
            yaxis=dict(
                title="Truth", dtick=1, zeroline=False, scaleanchor="x", scaleratio=1,
            ),
            xaxis=dict(
                title="Prediction",
                dtick=1,
                zeroline=False,
                range=to9,
                constrain="domain",
            ),
            margin={"l": 40, "b": 50, "r": 40, "t": 40},
            annotations=anno_text,
            title=dict(
                color="#000000",
                text="Confusion Matrix at Epoch {}".format(current_epoch),
            ),
            **{"height": 450, "width": 450},
        ),
    }
    acc_pt = accuData["points"][0]
    acc_pt["z"] = int(cf_data["text"][acc_pt["y"]][acc_pt["x"]])
    if acc_pt["z"] == 0:
        acc_pt["x"] = 0
        acc_pt["y"] = 0
        acc_pt["z"] = int(cf_data["text"][acc_pt["y"]][acc_pt["x"]])
    return ans, accuData


@app.callback(
    dd.Output("split-chart", "figure"),
    [dd.Input("accuracy-heatmap", "hoverData")],
    [
        dd.State("current-epoch", "children"),
        dd.State("splits-dict", "children"),
        dd.State("graph-colors", "children"),
    ],
)
def pie_splits(accuData, current_epoch, splits_dict, colors):
    epoch = int(current_epoch)
    pt = accuData["points"][0]
    if pt["z"] == 0:
        return dash.no_update
    sp_data = dict_from_string(splits_dict)[str(epoch)][pt["y"], pt["x"]]
    to9 = list(range(10))
    colors = dict_from_string(colors)["pie-splits"]
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
            yaxis=dict(scaleanchor="x", scaleratio=1),
            **{"height": 450, "width": 450},
        ),
    }


@app.callback(
    [dd.Output("testing-truth", "value"), dd.Output("testing-preds", "value")],
    [dd.Input("accuracy-heatmap", "hoverData")],
)
def testing_dropdowns(accuData):
    pt = accuData["points"][0]
    if pt["z"] == 0:
        return dash.no_update, dash.no_update
    return str(pt["y"]), str(pt["x"])


@app.callback(
    [
        dd.Output("network-error", "children"),
        dd.Output("gdd-images", "children"),
        dd.Output("gdd-scores", "children"),
        dd.Output("network-area", "style"),
    ],
    [dd.Input("grads-button", "n_clicks")],
    [
        dd.State("testing-truth", "value"),
        dd.State("testing-preds", "value"),
        dd.State("samples-dict", "children"),
        dd.State("current-mod", "children"),
        dd.State("current-run", "children"),
        dd.State("current-epoch", "children"),
    ],
)
def get_gradients(n_clicks, truth, pred, samples_dict, mod_name, run, epoch):
    t, p = int(truth), int(pred)
    samples_dict = dict_from_string(samples_dict)
    mod_name = Constants.regex_fix.sub("", dict_from_string(mod_name))
    run = Constants.regex_fix.sub("", dict_from_string(run))
    k = str(10 * t + p)
    samples = samples_dict[epoch].get(k, None)
    if samples is None:
        return (
            "No such samples exist",
            dash.no_update,
            dash.no_update,
            dict(display="none"),
        )
    run_dir = join(Constants.result_dir, mod_name, "runs", run)
    err_string = (
        "Should get grads for when truth is {} and pred is {}".format(truth, pred),
    )
    imgs, scores = get_grad_data(mod_name, run_dir, epoch, samples)
    return (
        "",
        dict_to_string(imgs),
        dict_to_string(scores),
        dict(),
    )


@app.callback(
    dd.Output("current-pred", "children"), [dd.Input("grad-scores", "hoverData")]
)
def current_prediction(scoreData):
    pt = scoreData["points"][0]
    return str(pt["y"])


@app.callback(
    dd.Output("grad-images", "figure"),
    [
        dd.Input("current-pred", "children"),
        dd.Input("network-area", "style"),
        dd.Input("grads-colorbar", "value"),
    ],
    [dd.State("gdd-images", "children"), dd.State("gdd-scores", "children")],
)
def gradient_images(current_pred, parent_style, color_range, gdd_images, gdd_scores):
    if parent_style.get("display", None) == "none":
        return {}
    gdd_images = dict_from_string(gdd_images)
    gdd_scores = dict_from_string(gdd_scores)
    inp, grad = (
        gdd_images["input"],
        gdd_images["grads"][int(current_pred)],
    )
    truth = gdd_scores["truth"]
    grad_colors = [
        [0, "rgba(5, 48, 97, 70)"],
        [0.44, "rgba(107, 172, 208, 20)"],
        [0.5, "rgba(248, 248, 248, 0)"],
        [0.56, "rgba(229, 130, 103, 20)"],
        [1, "rgba(103, 0, 31, 70)"],
    ]
    if color_range == "global":
        zmax = gdd_images["grads"].max()
        zmid = 0
    else:
        zmax = grad.max()
        zmid = 0 if 0 < zmax else (zmin + zmax) / 2
        if color_range == "none":
            grad_colors = [[0, "rgba(248,248,248,0)"], [1, "rgba(248,248,248,0)"]]
    to28 = list(range(28))
    return {
        "data": [
            go.Heatmap(
                x=to28,
                y=to28,
                z=inp[::-1],
                xgap=1,
                ygap=1,
                showscale=False,
                hoverongaps=False,
                colorscale="Greys",
                hovertemplate="<extra></extra>",
            ),
            go.Heatmap(
                x=to28,
                y=to28,
                z=grad[::-1],
                customdata=inp[::-1],
                xgap=1,
                ygap=1,
                zmid=0,
                zmin=-zmax,
                zmax=zmax,
                showscale=(color_range != "none"),
                hoverongaps=False,
                colorscale=grad_colors,
                opacity=0.9,
                hovertemplate="val = %{customdata} gradient = %{z}<extra></extra>",
            ),
        ],
        "layout": dict(
            hovermode="closest",
            font={"family": "et-book", "size": 15},
            width="600",
            height="600",
            yaxis=dict(visible=False),
            xaxis=dict(visible=False),
            margin={"l": 40, "b": 50, "r": 40, "t": 40},
            title=dict(
                color="#000000",
                text="Truth = {}, Prediction = {}".format(truth, current_pred),
            ),
        ),
    }


@app.callback(
    dd.Output("grad-scores", "figure"),
    [dd.Input("current-pred", "children"), dd.Input("network-area", "style")],
    [dd.State("gdd-scores", "children")],
)
def gradient_scores(current_pred, parent_style, gdd_scores):
    if parent_style.get("display", None) == "none":
        return {}
    gdd_scores = dict_from_string(gdd_scores)

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

    return {
        "data": [
            go.Heatmap(
                y=[0],
                xgap=1,
                x=list(range(10)),
                ygap=1,
                colorscale="Greens",
                z=gdd_scores["preds"].T,
                hovertemplate="<extra></extra>",
                showscale=False,
                hoverongaps=False,
                opacity=0.6,
            ),
        ],
        "layout": dict(
            hovermode="closest",
            title=dict(text="Scores"),
            font={"family": "et-book", "size": 15},
            annotations=[
                get_annotation(0.5, y, "%0.6f" % (gdd_scores["preds"][0, y]))
                for y in range(10)
            ],
            width="100",
            height="600",
            yaxis=dict(dtick=1, zeroline=False),
            xaxis=dict(visible=False,),
            margin={"l": 20, "b": 50, "r": 0, "t": 40},
        ),
    }


@app.server.route("/<path:im_path>.svg")
def net_struct(im_path):
    # download the file locally and then show it to user
    struct_path = join(Constants.result_dir, dirname(im_path), "network.svg")
    assert check_existence(struct_path), "Unable to access network svg"
    return send_from_directory(
        join(Constants.result_dir, dirname(im_path)), "network.svg"
    )


@app.callback(
    dd.Output("current-mod", "children"),
    [dd.Input("url", "pathname")],
    [dd.State("current-mod", "children")],
)
def cleanup_folder(pathname, mod_name):
    if pathname == "/":
        if mod_name != "" and not Constants.using_local_data:
            mod_name = dict_from_string(mod_name)
            rmtree(join(Constants.result_dir, mod_name))
        return ""
    return mod_name
