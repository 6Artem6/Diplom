from sitegpt.services.site_graph.page_graph_builder_with_events import PageGraphBuilderWithEvents


def test_page_graph_builder_with_events_creates_nodes():
    html = """
    <html>
        <head><script src='/static/vue.js'></script></head>
        <body>
            <form><input name="name" required></form>
        </body>
    </html>
    """
    builder = PageGraphBuilderWithEvents(html, "http://test.com")
    graph = builder.build()

    page_node = "page:http://test.com"
    assert page_node in graph.nodes
    data = graph.nodes[page_node]
    assert "frameworks" in data
    assert "complexity" in data
    assert "api_routes" in data
    assert "js_validations" in data
