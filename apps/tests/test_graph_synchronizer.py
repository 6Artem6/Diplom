import pytest
from sitegpt.services.graph_synchronizer import GraphSynchronizer
from sitegpt.services.embeddings.element_embedding import ElementSemanticEmbedding
from sitegpt.services.site_graph.page_graph_builder_with_events import (
    PageGraphBuilderWithEvents,
)
from sitegpt.services.action_graph.action_graph_builder import ActionGraphBuilder
from sitegpt.services.embeddings.vector_db import QdrantVectorDB


@pytest.fixture
def setup_services():
    vector_db = QdrantVectorDB(
        host="localhost", port=6333, collection_name="test_elements"
    )
    element_embedding_service = ElementSemanticEmbedding(vector_db)
    page_graph_service = PageGraphBuilderWithEvents()
    action_graph_service = ActionGraphBuilder()
    graph_sync = GraphSynchronizer(
        page_graph_service, element_embedding_service, action_graph_service
    )
    return (
        graph_sync,
        page_graph_service,
        element_embedding_service,
        action_graph_service,
    )


def test_sync_page_changes(setup_services):
    (
        graph_sync,
        page_graph_service,
        element_embedding_service,
        action_graph_service,
    ) = setup_services

    old_pg = page_graph_service.get_page_graph("home_old")
    new_pg = page_graph_service.get_page_graph("home_new")
    graph_sync.sync_page_changes(old_pg, new_pg)

    # Проверяем, что эмбеддинги созданы
    for elem in new_pg.elements:
        results = element_embedding_service.find_similar_elements(elem)
        assert results is not None

    # Проверяем, что связи с ActionGraph обновлены
    for elem in new_pg.elements:
        assert action_graph_service.get_actions_by_element(elem.element_id) is not None
