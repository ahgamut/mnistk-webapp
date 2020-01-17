import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies as dd
import dash_table as dtable
from plotly import graph_objects as go


def set_layout(app):
    def layout(pathname):
        return html.Div(
            children=[
                html.P("I should be showing data for {}".format(pathname)),
                dcc.Link("Go Back to Overview", href="/"),
            ],
            id="single-content",
        )

    return layout


def set_callbacks(app):
    pass
