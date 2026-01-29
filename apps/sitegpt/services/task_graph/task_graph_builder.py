import networkx as nx


class TaskGraphBuilder:
    def __init__(self, action_graph):
        """
        :param action_graph: граф действий (ActionGraphBuilder.graph)
        """
        self.action_graph = action_graph
        self.task_graph = nx.DiGraph()

    def build(self, user_request: str):
        """
        Строим граф задач на основе user_request и action_graph.
        Пока что делаем простое сопоставление: ищем ключевые слова.
        """
        self.task_graph.clear()

        if "логин" in user_request.lower():
            self._add_login_flow()

        if "перейди" in user_request.lower() or "next" in user_request.lower():
            self._add_navigation_flow()

        return self.task_graph

    def _add_login_flow(self):
        username_node = self._find_node_by_action("input:text")
        password_node = self._find_node_by_action("input:password")
        submit_node = self._find_node_by_action("click:button")

        if username_node and password_node and submit_node:
            self.task_graph.add_node(
                "task:username", action="input:text", selector=username_node
            )
            self.task_graph.add_node(
                "task:password", action="input:password", selector=password_node
            )
            self.task_graph.add_node(
                "task:submit", action="click:button", selector=submit_node
            )

            self.task_graph.add_edge("task:username", "task:password")
            self.task_graph.add_edge("task:password", "task:submit")

    def _add_navigation_flow(self):
        link_node = self._find_node_by_action("click:navigate")
        if link_node:
            self.task_graph.add_node(
                "task:navigate", action="click:navigate", selector=link_node
            )

    def _find_node_by_action(self, action_type):
        """
        Находит первый элемент в action_graph с нужным действием.
        """
        for node_id, data in self.action_graph.nodes(data=True):
            if data.get("action") == action_type:
                return node_id
        return None
