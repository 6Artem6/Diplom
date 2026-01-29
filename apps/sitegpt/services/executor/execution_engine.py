from typing import List, Dict, Any


class ExecutionEngine:
    def __init__(self, browser_adapter, trace_logger):
        self.browser = browser_adapter
        self.logger = trace_logger

    async def execute_sequence(
        self, action_sequence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        trace = {"steps": [], "success": True}

        for idx, action in enumerate(action_sequence):
            step_result = {"step": idx, "action": action}
            try:
                result = await self._execute_action(action)
                step_result["result"] = result

                # сохраняем снапшот после шага
                snapshot = await self.browser.snapshot(f"step_{idx}")
                step_result.update(snapshot)

                step_result["status"] = "ok"
            except Exception as e:
                step_result["status"] = "error"
                step_result["error"] = str(e)
                trace["success"] = False
                break
            trace["steps"].append(step_result)

        self.logger.log(trace)
        return trace

    async def _execute_action(self, action: Dict[str, Any]):
        action_type = action.get("action")
        selector = action.get("selector")
        value = action.get("value")

        if action_type == "click:navigate":
            return await self.browser.click(selector)
        elif action_type.startswith("click"):
            return await self.browser.click(selector)
        elif action_type.startswith("input"):
            return await self.browser.input_text(selector, value or "")
        elif action_type.startswith("submit"):
            return await self.browser.submit_form(selector)
        elif action_type == "choose:option":
            return await self.browser.choose_option(selector, value)
        elif action_type == "navigate":
            return await self.browser.navigate(action["url"])
        else:
            raise ValueError(f"Unknown action: {action_type}")
