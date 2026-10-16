import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from graph.builder import _on_update_user_profile_error
from memory.semantic import make_semantic_memory_tool


class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _run(update_args: dict, with_middleware: bool):
    store = InMemoryStore()
    script = iter([
        AIMessage(content="", tool_calls=[{
            "name": "update_user_profile",
            "args": {"content": {"price_max": 200000}, "action": "update", **update_args},
            "id": "call-1",
        }]),
        AIMessage(content="done"),
    ])
    middleware = (
        [ToolErrorMiddleware(_on_update_user_profile_error, tools=["update_user_profile"])]
        if with_middleware else []
    )
    agent = create_agent(
        model=_ScriptedModel(messages=script),
        tools=[make_semantic_memory_tool(1, store)],
        middleware=middleware,
        store=store,
    )
    result = agent.invoke({"messages": [HumanMessage("hi")]})
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    return result["messages"][-1].content, tool_msgs[0].content


@pytest.mark.parametrize("with_middleware", [True, False])
def test_empty_id_is_handled_by_the_framework_either_way(with_middleware):
    final, tool_text = _run({"id": ""}, with_middleware)
    assert final == "done"
    assert "Input should be a valid UUID" in tool_text


def test_omitted_id_aborts_the_run_without_middleware():
    with pytest.raises(ValueError, match="MEMORY ID"):
        _run({}, with_middleware=False)


def test_omitted_id_is_recovered_with_middleware():
    final, tool_text = _run({}, with_middleware=True)
    assert final == "done"
    assert tool_text.startswith("Profile not saved")
    assert "action='create'" in tool_text


def test_handler_ignores_non_value_errors():
    assert _on_update_user_profile_error(RuntimeError("boom"), request=None) is None
