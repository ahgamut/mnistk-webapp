from app_display.paging import PageHandler


def view(result_dir):
    """
    View results in a Dash webapp
    """
    ph = PageHandler(result_dir)
    ph.app.run_server(debug=True, dev_tools_hot_reload=False)


if __name__ == "__main__":
    view("./results/")
