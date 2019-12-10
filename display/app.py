from overview import set_app_layout
import dash
import pandas as pd


def get_application(csv_path):
    df = pd.read_csv(csv_path, header=0)
    external_css = ["https://cdnjs.cloudflare.com/ajax/libs/tufte-css/1.7.2/tufte.css"]
    app = dash.Dash(__name__, external_stylesheets=external_css)
    app.config.suppress_callback_exceptions = True
    set_app_layout(df, app)
    return app


def view(result_dir, csv_path):
    """
    View results in a Dash webapp
    """
    app = get_application(csv_path)
    app.run_server(debug=True, dev_tools_hot_reload=False)


if __name__ == "__main__":
    view("../results/", "../results/summary.csv")
