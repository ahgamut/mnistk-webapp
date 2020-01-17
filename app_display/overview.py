# -*- coding: utf-8 -*-
"""
    mnistk.display.overview
    ~~~~~~~~~~~~~~~~~~~~~~~

    Show overview of network performance via Dash

    :copyright: (c) 2020 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies as dd
import dash_table as dtable
from plotly import graph_objects as go


def lv(label, value, vlabel=None):
    vlab = label if vlabel is None else vlabel
    return {"label": label, "value": "{}|{}".format(vlab, value)}


def halved_div(func, div_string="", split=50):
    div = html.Div(
        children=[
            html.Div(
                div_string, style=dict(width="{}%".format(split), display="table-cell")
            ),
            html.Div(
                func, style=dict(width="{}%".format(100 - split), display="table-cell")
            ),
        ],
        style=dict(display="table", margin=10),
        className="fullwidth",
    )
    return div


def filter_options(df):
    filter_opts = []
    for (r, e), _ in df.groupby(["run", "epoch"]):
        filter_opts.append(lv("run {}, epoch {}".format(r, e), "{}|{}".format(r, e)))
    default = filter_opts[0]["value"]
    dropdown = dcc.Dropdown(
        id="filter-options",
        options=filter_opts,
        multi=True,
        value=[],
        placeholder="Showing all runs; select run(s) here",
    )
    return halved_div(dropdown, "Select Run(s) to view:", 30)


def grouping_options():
    group_opts = [
        lv("Network Style", "groupname"),
        lv("Activation Function", "activation"),
        lv("Epoch", "epoch"),
    ]
    default = group_opts[0]["value"]
    dropdown = dcc.Dropdown(id="grouping-options", options=group_opts, value=default)
    return halved_div(dropdown, "Group Networks By:", 30)


def xvalue_options():
    xval_opts = [
        lv("Time taken to train (s)", "time"),
        lv("Number of Parameters used", "#params"),
        lv("Approx. Memory Usage (bytes)", "memory per pass"),
        lv("Number of Layers", "#layers"),
        lv("Number of Operations", "#ops"),
    ]
    default = xval_opts[0]["value"]
    dropdown = dcc.Dropdown(
        id="perfx-dropdown", options=xval_opts, value=default, clearable=False
    )
    range_measure = dcc.RangeSlider(
        id="perfx-range",
        allowCross=False,
        min=0,
        max=100,
        step=1,
        marks={0: "Min", 100: "Max"},
        value=[0, 100],
    )
    div0 = halved_div(dropdown, "X-Axis Measures:", 30)
    div1 = halved_div(range_measure, "", 30)
    return html.Div([div0, div1])


def yvalue_options():
    yval_opts = (
        [lv("Overall Accuracy", "accuracy"), lv("Overall AUC", "AUC")]
        + [
            lv(
                "{} pred. Accuracy".format(x),
                "accuracy_{}".format(x),
                "% correct {} predictions".format(x),
            )
            for x in range(10)
        ]
        + [
            lv(
                "{} pred. AUC".format(x),
                "AUC_{}".format(x),
                "AUC when predicting {}".format(x),
            )
            for x in range(10)
        ]
    )

    default = yval_opts[0]["value"]
    dropdown = dcc.Dropdown(
        id="perfy-dropdown", options=yval_opts, value=default, clearable=False
    )
    range_measure = dcc.RangeSlider(
        id="perfy-range",
        allowCross=False,
        min=0,
        max=100,
        step=1,
        marks={0: "Min", 100: "Max"},
        value=[0, 100],
    )
    div0 = halved_div(dropdown, "Y-Axis Measures:", 30)
    div1 = halved_div(range_measure, "", 30)
    return html.Div([div0, div1])


class DataHandler(object):
    df_full = None
    df_subset = None
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
    measure_columns = (
        ["accuracy", "AUC"]
        + ["accuracy_{}".format(i) for i in range(10)]
        + ["AUC_{}".format(i) for i in range(10)]
    )

    def __init__(self, df_full):
        DataHandler.df_full = df_full
        option_string = self.subsetting()

    @staticmethod
    def subsetting(
        filter_opt=[],
        group_opt="Network Style|groupname",
        xval="Time taken|time",
        x_range=[0, 100],
        yval="Overall Accuracy|accuracy",
        y_range=[0, 100],
    ):
        df_full = DataHandler.df_full
        if isinstance(filter_opt, str):
            filter_opt = [filter_opt]
        if len(filter_opt) == 0:
            df = df_full
        else:
            filter_opt = [x.split("|")[1:] for x in filter_opt]
            runs = set([x[0] for x in filter_opt])
            epochs = set([int(x[1]) for x in filter_opt])
            df = df_full[
                (
                    df_full["run"].apply(lambda x: x in runs)
                    & df_full["epoch"].apply(lambda x: x in epochs)
                )
            ]

        xtitle, xcol = xval.split("|")
        ytitle, ycol = yval.split("|")
        gp_title, gp_opt = group_opt.split("|")
        xmin, xmax = min(df[xcol]), max(df[xcol])
        xleft = xmin + (xmax - xmin) * (0.01) * (x_range[0])
        xright = xmin + (xmax - xmin) * (0.01) * (x_range[1])

        ymin, ymax = min(df[ycol]), max(df[ycol])
        yleft = ymin + (ymax - ymin) * (0.01) * (y_range[0])
        yright = ymin + (ymax - ymin) * (0.01) * (y_range[1])

        df = df[
            (df[xcol] >= xleft)
            & (df[ycol] >= yleft)
            & (df[xcol] <= xright)
            & (df[ycol] <= yright)
        ]
        DataHandler.df_subset = df
        parsed_options = dict(
            xtitle=xtitle,
            xcol=xcol,
            ytitle=ytitle,
            ycol=ycol,
            gp_title=gp_title,
            gp_opt=gp_opt,
        )
        return json.dumps(parsed_options)

    @staticmethod
    def top10table(option_string):
        df = DataHandler.df_subset
        options = json.loads(option_string)
        xtitle, xcol = options["xtitle"], options["xcol"]
        ytitle, ycol = options["ytitle"], options["ycol"]

        df2 = df.sort_values(by=[ycol, xcol], ascending=[False, True]).head(10).round(6)
        important = set([ycol, "accuracy", "AUC"])
        col_order = (
            ["name", "run", "epoch", "time", "#params", "memory per pass"]
            + list(sorted(set(important)))
            + list(sorted(set(DataHandler.measure_columns) - important))
        )
        dt = (
            dtable.DataTable(
                id="top10-table",
                columns=[{"name": col, "id": col} for col in col_order],
                data=df2.to_dict("records"),
                style_table={"overflowX": "scroll"},
                style_cell={"overflow": "hidden", "textOverflow": "ellipsis"},
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                    "font-family": "et-book",
                    "font-size": 20,
                },
                fill_width=False,
                style_data_conditional=[
                    {
                        "if": dict(
                            column_id=x, filter_query="{%s} ge %.6f" % (x, max(df2[x]))
                        ),
                        "fontWeight": "bold",
                        "backgroundColor": "rgb(102,255,51)",
                    }
                    for x in DataHandler.measure_columns
                ]
                + [
                    {
                        "if": dict(
                            column_id=x, filter_query="{%s} le %.6f" % (x, min(df2[x]))
                        ),
                        "font-style": "italic",
                        "backgroundColor": "rgb(102,255,51)",
                    }
                    for x in ["time", "#params", "memory per pass"]
                ],
            ),
        )
        return dt

    @staticmethod
    def figure(option_string):
        df = DataHandler.df_subset
        options = json.loads(option_string)
        gp_title, gp_opt = options["gp_title"], options["gp_opt"]
        xtitle, xcol = options["xtitle"], options["xcol"]
        ytitle, ycol = options["ytitle"], options["ycol"]
        ans = {
            "data": [
                go.Scatter(
                    x=gp[xcol],
                    y=gp[ycol],
                    mode="markers",
                    marker=dict(size=12),
                    opacity=0.6,
                    name=gname,
                    text=gp["name"],
                    customdata="run "
                    + gp["run"]
                    + ", epoch "
                    + gp["epoch"].astype(str),
                    hovertemplate="(x=%{x}, y=%{y})<br>%{text}<extra>%{customdata}</extra>",
                )
                for gname, gp in df.groupby(gp_opt)
            ],
            "layout": dict(
                hovermode="closest",
                transition=dict(duration=700),
                xaxis={"title": xtitle},
                yaxis={"title": ytitle},
                height="700",
                font={"family": "et-book", "size": 15},
                clickmode="event",
            ),
        }
        return ans

    @staticmethod
    def data_select(clickData):
        if clickData is None:
            return dash.no_update
        pt = clickData["points"][0]
        run, ep = (
            pt["customdata"]
            .replace("run", "")
            .replace("epoch", "")
            .replace(" ", "")
            .split(",")
        )
        return "/{}/{}/{}".format(pt["text"], run, ep)


def set_callbacks(df, app):
    dh = DataHandler(df)
    inputs = [
        dd.Input(component_id="filter-options", component_property="value"),
        dd.Input(component_id="grouping-options", component_property="value"),
        dd.Input(component_id="perfx-dropdown", component_property="value"),
        dd.Input(component_id="perfx-range", component_property="value"),
        dd.Input(component_id="perfy-dropdown", component_property="value"),
        dd.Input(component_id="perfy-range", component_property="value"),
    ]
    app.callback(
        dd.Output(component_id="compute-div1", component_property="children"), inputs
    )(DataHandler.subsetting)
    app.callback(
        dd.Output(component_id="perf-graph", component_property="figure"),
        [dd.Input(component_id="compute-div1", component_property="children")],
    )(DataHandler.figure)
    app.callback(
        dd.Output(component_id="top10-div", component_property="children"),
        [dd.Input(component_id="compute-div1", component_property="children")],
    )(DataHandler.top10table)
    app.callback(
        dd.Output(component_id="url", component_property="pathname"),
        [dd.Input(component_id="perf-graph", component_property="clickData")],
    )(DataHandler.data_select)


def set_layout(df, app):
    x_select = xvalue_options()
    y_select = yvalue_options()
    filt_select = filter_options(df)
    gp_select = grouping_options()

    layout = html.Div(
        children=[
            html.Div(html.P("Some talk here about what this page is")),
            html.H2("View the performance of a thousand neural networks"),
            html.Div(id="compute-div1", children="options", style={"display": "none"}),
            dcc.Graph(
                id="perf-graph",
                className="fullwidth",
                config=dict(displayModeBar=False),
            ),
            html.P([x_select, y_select, filt_select, gp_select]),
            html.Div(html.P("", id="figure-click")),
            html.H2("The Top Ten"),
            html.Div(id="top10-div", className="fullwidth"),
        ],
        id="overview-content",
    )
    return layout
