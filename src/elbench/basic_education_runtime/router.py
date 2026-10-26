from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.messages import BaseMessage
from typing import Union, Sequence, Callable, Tuple, Dict

from elbench.basic_education_runtime.utils import content_to_text, remove_think


def any_keyword_route(
    keywords: Sequence[str], exists_to: str, else_to: str, think_as_message: bool = False
) -> Tuple[Callable[..., bool], Dict[bool, str]]:
    """Route based on keywords."""
    path_map = {
        True: exists_to,
        False: else_to,
    }

    def route(state: Union[AgentState, Sequence[BaseMessage]]) -> bool:
        """Route based on keywords."""
        if isinstance(state, Sequence):
            message = state[-1]
        elif messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No messages found, error while routing")

        content = message.content
        if not think_as_message:
            content = remove_think(content)  # Remove think tags
        text = content_to_text(content, include_reasoning=think_as_message).lower()
        return any(keyword in text for keyword in keywords)

    return (route, path_map)


def all_keyword_route(
    keywords: Sequence[str], exists_to: str, else_to: str
) -> Tuple[Callable[..., bool], Dict[bool, str]]:
    """Route based on keywords."""
    path_map = {
        True: exists_to,
        False: else_to,
    }

    def route(state: Union[AgentState, Sequence[BaseMessage]]) -> bool:
        """Route based on keywords."""
        if isinstance(state, Sequence):
            message = state[-1]
        elif messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No messages found, error while routing")

        text = content_to_text(message.content).lower()
        return all(keyword in text for keyword in keywords)

    return (route, path_map)

