from sitegpt.services.page_graph.page_graph_builder_with_events import (
    PageGraphBuilderWithEvents,
)
from sitegpt.services.embeddings.element_embedding import ElementSemanticEmbedding
from sitegpt.services.action_graph.action_graph_builder import ActionGraphBuilder


class GraphSynchronizer:
    def __init__(
        self,
        page_graph_service: PageGraphBuilderWithEvents,
        element_embedding_service: ElementSemanticEmbedding,
        action_graph_service: ActionGraphBuilder,
    ):
        self.page_graph_service = page_graph_service
        self.element_embedding_service = element_embedding_service
        self.action_graph_service = action_graph_service

    def sync_page_changes(self, old_page_graph, new_page_graph):
        updated_elements = self.detect_changed_elements(old_page_graph, new_page_graph)
        self.page_graph_service.update_graph(new_page_graph)

        if updated_elements:
            batch_items = [
                {"element": elem, "node_id": elem.element_id}
                for elem in updated_elements
            ]
            self.element_embedding_service.batch_create_embeddings(batch_items)

        self.recompute_action_mappings(updated_elements)
        self.recompute_semantic_links(updated_elements)

    def detect_changed_elements(self, old_pg, new_pg):
        old_elements = {e.element_id: e for e in old_pg.elements}
        new_elements = {e.element_id: e for e in new_pg.elements}

        changed = []
        for node_id, elem in new_elements.items():
            if node_id not in old_elements or self.has_element_changed(
                old_elements[node_id], elem
            ):
                changed.append(elem)
        return changed

    @staticmethod
    def has_element_changed(old_elem, new_elem):
        return (
            old_elem.text_content != new_elem.text_content
            or old_elem.placeholder != new_elem.placeholder
            or set(old_elem.css_classes) != set(new_elem.css_classes)
            or old_elem.aria_label != new_elem.aria_label
        )

    def recompute_action_mappings(self, updated_elements):
        self.action_graph_service.map_elements_to_actions(updated_elements)

    def recompute_semantic_links(self, updated_elements):
        # Placeholder: сюда можно добавить cross-modal links
        pass
