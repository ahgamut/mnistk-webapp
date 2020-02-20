# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies as dd
import dash_table as dtable
from plotly import graph_objects as go
import sqlalchemy as sa
from pandas import read_sql_query
from .textinfo import intro_text_0, top10_text_0, intro_text_1, ov_text_0
from .utils import lv, check_existence, dict_from_string, dict_to_string
from app import app, Constants


def halved_div(func, func2="", split=50, **kwargs):
    div = html.Div(
        children=[
            html.Div(
                func2, style=dict(width="{}%".format(split), display="table-cell")
            ),
            html.Div(
                func, style=dict(width="{}%".format(100 - split), display="table-cell")
            ),
        ],
        style=dict(display="table", margin=10),
        **kwargs,
    )
    return div


####################
# Layout functions #
####################


def snapshot_options():
    snapshot_opts = []
    engine = sa.create_engine("sqlite:///{}".format(Constants.db_path))
    df = read_sql_query(
        sa.select([sa.column("run"), sa.column("epoch")])
        .distinct()
        .select_from(sa.table("summary")),
        engine,
    )
    for i in range(len(df)):
        r = df.loc[i, "run"]
        e = df.loc[i, "epoch"]
        snapshot_opts.append(lv("run {}, epoch {}".format(r, e), "{}|{}".format(r, e)))
    default = snapshot_opts[0]["value"]
    dropdown = dcc.Dropdown(
        id="snapshot-options",
        options=snapshot_opts,
        multi=True,
        value=[],
        placeholder="Showing all runs; select run(s) here",
    )
    return halved_div(dropdown, "Select Run(s) to view:", 30, className="fullwidth")


def groupby_options():
    gb_opts = [
        lv("Network Class", "groupname"),
        lv("Network Subclass", "formname"),
        lv("Activation Function", "activation"),
        lv("Epoch", "epoch"),
    ]
    default = gb_opts[0]["value"]
    dropdown = dcc.Dropdown(
        id="groupby-options", options=gb_opts, value=default, clearable=False
    )
    return halved_div(dropdown, "Group Networks By:", 30, className="fullwidth")


def subset_options():
    subset_opts = []
    engine = sa.create_engine("sqlite:///{}".format(Constants.db_path))
    df = read_sql_query(
        sa.select([sa.column("groupname")]).distinct().select_from(sa.table("summary")),
        engine,
    )
    subset_opts = [{"label": k, "value": k} for k in df["groupname"].tolist()]
    # default = subset_opts[0]["value"]
    dropdown = dcc.Dropdown(
        id="subset-options",
        options=subset_opts,
        multi=True,
        value=[],
        placeholder="Showing all classes; select classes here",
    )
    return halved_div(dropdown, "Select Classes to view:", 30, className="fullwidth")


def get_ranges():
    range_dict = {}
    engine = sa.create_engine("sqlite:///{}".format(Constants.db_path))
    for colstr in Constants.x_columns + Constants.y_columns:
        col = sa.column(colstr)
        query = sa.select([sa.func.min(col), sa.func.max(col)]).select_from(
            sa.table("summary")
        )
        a, b = read_sql_query(query, engine).iloc[0, :]
        range_dict[colstr] = [a, b]
    return range_dict


def xvalue_options():
    xval_opts = [
        lv("Time taken to train (s)", "time"),
        lv("Number of Parameters used", "#params"),
        lv("Approx. Memory Usage (bytes)", "memory per pass"),
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
    div0 = halved_div(dropdown, "X-Axis Measures:", 30, className="fullwidth")
    div1 = halved_div(range_measure, "", 30, className="fullwidth")
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
        marks={0: "0", 100: "1"},
        value=[0, 100],
    )
    div0 = halved_div(dropdown, "Y-Axis Measures:", 30, className="fullwidth")
    div1 = halved_div(range_measure, "", 30, className="fullwidth")
    return html.Div([div0, div1])


def set_layout():
    x_select = xvalue_options()
    y_select = yvalue_options()
    snap_select = snapshot_options()
    gp_select = groupby_options()
    sub_select = subset_options()

    layout = html.Div(
        children=[
            html.H2("Introduction"),
            dcc.Markdown(intro_text_0, style=dict(width="75%")),
            html.H2("View the performance of a thousand neural networks"),
            html.Div(
                dict_to_string(get_ranges()),
                id="ranges-div",
                style=dict(display="none"),
            ),
            dcc.Markdown(ov_text_0, style=dict(width="75%")),
            dcc.Graph(
                id="perf-graph",
                className="fullwidth",
                config=dict(displayModeBar=False),
            ),
            html.P([x_select, y_select, gp_select, sub_select, snap_select]),
            dcc.Markdown(intro_text_1, style=dict(width="75%")),
            html.H2("The Top Ten"),
            dcc.Markdown(top10_text_0, style=dict(width="75%")),
            html.Div(id="top10-div", className="fullwidth"),
        ],
        id="overview-content",
    )
    return layout


######################
# Callback functions #
######################


def top10table(expr, options):
    xtitle, xcol = options["xtitle"], options["xcol"]
    ytitle, ycol = options["ytitle"], options["ycol"]

    df_clause = (
        sa.select(["*"])
        .select_from(sa.table("summary"))
        .where(expr)
        .order_by(sa.desc(sa.column(ycol)))
        .limit(10)
    )
    engine = sa.create_engine("sqlite:///{}".format(Constants.db_path))
    df = read_sql_query(df_clause, engine)
    df2 = df.sort_values(by=[ycol, xcol], ascending=[False, True]).round(6)
    important = set([ycol, "accuracy", "AUC"])
    col_order = (
        ["name", "run", "epoch"]
        + Constants.x_columns
        + list(sorted(set(important)))
        + list(sorted(set(Constants.y_columns) - important))
    )
    dt = (
        dtable.DataTable(
            id="top10-table",
            columns=[
                {"name": Constants.pretty_names.get(col, col), "id": col}
                for col in col_order
            ],
            data=df2.to_dict("records"),
            row_selectable="single",
            style_table={"overflowX": "scroll"},
            style_cell={
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "font-family": "et-book",
                "font-size": 20,
            },
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
                        column_id=x, filter_query="{%s} ge %.6f" % (x, max(df2[x])),
                    ),
                    "fontWeight": "bold",
                    "backgroundColor": "rgb(102,255,51)",
                }
                for x in Constants.y_columns
            ]
            + [
                {
                    "if": dict(
                        column_id=x, filter_query="{%s} le %.6f" % (x, min(df2[x])),
                    ),
                    "font-style": "italic",
                    "backgroundColor": "rgb(102,255,51)",
                }
                for x in Constants.x_columns
            ],
        ),
    )
    return dt


def figure(expr, options):
    gp_title, gp_opt = options["gp_title"], options["gp_opt"]
    xtitle, xcol = options["xtitle"], options["xcol"]
    ytitle, ycol = options["ytitle"], options["ycol"]

    engine = sa.create_engine("sqlite:///{}".format(Constants.db_path))
    required_cols = set(["name", "run", "epoch", gp_opt, xcol, ycol])
    df_clause = (
        sa.select([sa.column(x) for x in required_cols])
        .select_from(sa.table("summary"))
        .where(expr)
    )
    df = read_sql_query(df_clause, engine)
    ans = {
        "data": [
            go.Scattergl(
                x=gp[xcol],
                y=gp[ycol],
                mode="markers",
                marker=dict(size=12),
                opacity=0.6,
                name=gname,
                text=gp["name"],
                customdata="run " + gp["run"] + ", epoch " + gp["epoch"].astype(str),
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
            legend=dict(bgcolor="#fffff8"),
            clickmode="event",
        ),
    }
    return ans


@app.callback(
    [
        dd.Output(component_id="perf-graph", component_property="figure"),
        dd.Output(component_id="top10-div", component_property="children"),
    ],
    [
        dd.Input(component_id="snapshot-options", component_property="value"),
        dd.Input(component_id="subset-options", component_property="value"),
        dd.Input(component_id="groupby-options", component_property="value"),
        dd.Input(component_id="perfx-dropdown", component_property="value"),
        dd.Input(component_id="perfx-range", component_property="value"),
        dd.Input(component_id="perfy-dropdown", component_property="value"),
        dd.Input(component_id="perfy-range", component_property="value"),
    ],
    [dd.State("ranges-div", "children")],
)
def subsetting(
    snapshot_opt, subset_opt, gb_opt, xval, x_range, yval, y_range, ranges_str
):
    if isinstance(snapshot_opt, str):
        snapshot_opt = [snapshot_opt]

    snap_expr = sa.true()
    sub_expr = sa.true()
    run = sa.column("run")
    epoch = sa.column("epoch")

    if len(snapshot_opt) == 0:
        snap_expr = sa.true()
    else:
        snapshot_opt = [x.split("|")[1:] for x in snapshot_opt]
        snap_expr = sa.false()
        for r, e in snapshot_opt:
            snap_expr = snap_expr | ((run == r) & (epoch == int(e)))

    if len(subset_opt) == 0:
        sub_expr = sa.true()
    else:
        sub_expr = sa.column("groupname").in_(subset_opt)

    expr = (snap_expr) & (sub_expr)
    # expr is an sqlalchemy expression that will end up inside a where
    xtitle, xcol = xval.split("|")
    ytitle, ycol = yval.split("|")
    gp_title, gp_opt = gb_opt.split("|")

    range_dict = dict_from_string(ranges_str)
    xmin, xmax = range_dict[xcol]
    xleft = xmin + (xmax - xmin) * (0.01) * (x_range[0])
    xright = xmin + (xmax - xmin) * (0.01) * (x_range[1])

    ymin, ymax = range_dict[ycol]
    yleft = ymin + (ymax - ymin) * (0.01) * (y_range[0])
    yright = ymin + (ymax - ymin) * (0.01) * (y_range[1])

    xval = sa.column(xcol)
    yval = sa.column(ycol)
    expr = (expr) & xval.between(xleft, xright) & yval.between(yleft, yright)
    parsed_options = dict(
        xtitle=xtitle,
        xcol=xcol,
        ytitle=ytitle,
        ycol=ycol,
        gp_title=gp_title,
        gp_opt=gp_opt,
    )

    fig = figure(expr, parsed_options)
    table = top10table(expr, parsed_options)
    return fig, table


@app.callback(
    dd.Output(component_id="url", component_property="pathname"),
    [
        dd.Input(component_id="perf-graph", component_property="clickData"),
        dd.Input("top10-table", "selected_rows"),
    ],
    [dd.State("top10-table", "data")],
)
def data_select(clickData, row_index, table_data):
    if clickData is None and row_index is None:
        return dash.no_update
    elif clickData is not None:
        pt = clickData["points"][0]
        run, ep = (
            pt["customdata"]
            .replace("run", "")
            .replace("epoch", "")
            .replace(" ", "")
            .split(",")
        )
        mod_name = pt["text"]
    elif row_index is not None:
        row = table_data[row_index[0]]
        mod_name = row["name"]
        run = row["run"]
        ep = row["epoch"]

    mod_name = Constants.regex_fix.sub("", mod_name)
    run = Constants.regex_fix.sub("", run)
    ep = Constants.regex_fix.sub("", str(ep))
    return "/{}/{}/{}".format(mod_name, run, ep)


assert check_existence(Constants.db_path), "No data to show"
