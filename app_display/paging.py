import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies as dd
from .overview import set_layout as overview_layout, set_callbacks as overview_callbacks
from .singleton import set_layout as single_layout, set_callbacks as single_callbacks
import sys, os


class PageHandler(object):
    app = None
    result_dir = None

    def __init__(self, result_dir):
        PageHandler.result_dir = result_dir
        external_css = [
            "https://cdnjs.cloudflare.com/ajax/libs/tufte-css/1.7.2/tufte.css"
        ]
        PageHandler.app = dash.Dash("app", external_stylesheets=external_css)
        PageHandler.app.config.suppress_callback_exceptions = True
        PageHandler.app.layout = html.Div(
            children=[
                dcc.Location(id="url", refresh=True),
                html.H1(["mnistk - 1001 generated networks on MNIST"]),
                html.Div(id="page-content"),
            ]
        )
        setup_pages()
        PageHandler.app.callback(
            dd.Output(component_id="page-content", component_property="children"),
            [dd.Input(component_id="url", component_property="pathname")],
        )(display_page)


def setup_pages():
    overview_callbacks(
        os.path.join(PageHandler.result_dir, "summary.db"), PageHandler.app
    )
    single_callbacks(PageHandler.result_dir, PageHandler.app)


def display_page(pathname):
    if pathname == "/":
        return overview_layout()
    else:
        return single_layout(pathname)
