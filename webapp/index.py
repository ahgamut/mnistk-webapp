import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies as dd
from app import app, Constants
from app_display.overview import set_layout as overview_layout
from app_display.singleton import set_layout as single_layout

app.layout = html.Div(
    children=[
        dcc.Location(id="url", refresh=True),
        html.H1(["mnistk - 1001 generated networks on MNIST"]),
        html.Div(id="page-content"),
    ]
)
app.title = "mnistk - 1001 generated networks on MNIST"


@app.callback(
    dd.Output("page-content", "children"), [dd.Input("url", "pathname")],
)
def display_page(pathname):
    if pathname == "/":
        return overview_layout()
    else:
        return single_layout(pathname)


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_hot_reload=False)
