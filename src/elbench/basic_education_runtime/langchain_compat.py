"""Compatibility helpers for LangChain version differences."""

try:
    from langchain.globals import set_debug
except ModuleNotFoundError:
    try:
        from langchain_core.globals import set_debug
    except ModuleNotFoundError:
        def set_debug(_: bool) -> None:
            """Fallback when LangChain debug globals are unavailable."""

            return None
