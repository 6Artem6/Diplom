import re


class QueryAnalyzer:
    def analyze(self, query: str) -> dict:
        """
        Простейший NLU: ищем намерения и сущности в запросе.
        """
        query = query.lower()
        intent = None
        steps = []
        entities = {}

        if "логин" in query or "войти" in query:
            intent = "login"
            steps.append("login")
            entities["username"] = None
            entities["password"] = None

        if "перейди" in query or "next" in query or "страницу" in query:
            steps.append("navigate")
            if intent:
                intent = f"{intent}_and_navigate"
            else:
                intent = "navigate"

        # вычленим имя пользователя (например: "логинься как admin")
        m = re.search(r"как (\w+)", query)
        if m:
            entities["username"] = m.group(1)

        return {
            "intent": intent,
            "steps": steps,
            "entities": entities,
        }
