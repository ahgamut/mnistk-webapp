import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies as dd
from .overview import set_layout as ov_layout, set_callbacks as ov_callbacks
from .singleton import SPHandler
import pandas as pd
import sys, os


class PageHandler(object):
    app = None
    df = None
    overview_page = None
    single_page = None
    result_dir = None

    def __init__(self, result_dir):
        PageHandler.result_dir = result_dir
        csv_path = os.path.join(result_dir, "summary.csv")
        PageHandler.df = pd.read_csv(csv_path, header=0)
        external_css = [
            "https://cdnjs.cloudflare.com/ajax/libs/tufte-css/1.7.2/tufte.css"
        ]
        PageHandler.app = dash.Dash("app", external_stylesheets=external_css)
        PageHandler.app.config.suppress_callback_exceptions = True
        PageHandler.app.layout = html.Div(
            children=[
                dcc.Location(id="url", refresh=False),
                html.H1(["mnistk - 1001 generated networks on MNIST"]),
                html.P("some nice subtitle text here", className="subtitle"),
                html.Div(id="page-content"),
            ]
        )
        PageHandler.setup_pages()
        PageHandler.app.callback(
            dd.Output(component_id="page-content", component_property="children"),
            [dd.Input(component_id="url", component_property="pathname")],
        )(PageHandler.display_page)

    @staticmethod
    def setup_pages():
        PageHandler.overview_page = ov_layout(PageHandler.df, PageHandler.app)
        PageHandler.single_page = SPHandler(PageHandler.result_dir, PageHandler.app)
        ov_callbacks(PageHandler.df, PageHandler.app)

    @staticmethod
    def display_page(pathname):
        if pathname == "/":
            return PageHandler.overview_page
        else:
            return PageHandler.single_page.layout(pathname)
