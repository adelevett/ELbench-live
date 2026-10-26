import re
import asyncio
from typing import Any, Dict
from langchain_core.tools import tool, BaseTool
from elbench.basic_education_runtime.config import CONFIG
from langgraph.prebuilt import create_react_agent
from langchain.chat_models.base import BaseChatModel
from elbench.basic_education_runtime.entity import ExportFormat
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed
import json


def generate_evaluation_tool() -> BaseTool:
    """
    Generates a tool for evaluating the basic education runtime outputs.
    This function is a placeholder and should be implemented with actual logic.
    """

    if CONFIG.evaluation is None:
        raise ValueError("Evaluation configuration not found.")

    @tool(
        name_or_callable="save_result_to_database",
        description="Save the evaluation results to a database.",
        return_direct=True,
        args_schema=CONFIG.evaluation.format_to_pydantic(),
    )
    def save_to_db(**kwargs):
        """
        Save the evaluation results to a database.
        """
        for k, v in kwargs.items():
            if issubclass(v.__class__, BaseModel):
                v = v.model_dump()
            kwargs[k] = v
        return kwargs

    return save_to_db


def _build_evaluation_messages(
    *,
    system_prompt: str,
    other_prompts: list[Any],
    exported_result: ExportFormat,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for op in other_prompts:
        content = exported_result.replace_template(op.content)
        role = str(op.role or "user").strip() or "user"
        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    has_user_query = any(
        item.get("role") == "user" and str(item.get("content", "")).strip()
        for item in messages
    )
    if not has_user_query:
        fallback = exported_result.replace_template(system_prompt).strip()
        if not fallback:
            fallback = "Please complete the evaluation."
        messages.append(
            {
                "role": "user",
                "content": fallback,
            }
        )
    return messages


@retry(
    stop=stop_after_attempt(CONFIG.globals.retry.attempt),
    wait=wait_fixed(CONFIG.globals.retry.interval),
)
async def evaluate(
    model: BaseChatModel, exported_result: ExportFormat
) -> Dict[str, Any]:
    assert CONFIG.evaluation
    system_prompt, other_prompt = CONFIG.evaluation.get_prompts()
    system_prompt = exported_result.replace_template(system_prompt)
    ops = _build_evaluation_messages(
        system_prompt=system_prompt,
        other_prompts=other_prompt,
        exported_result=exported_result,
    )
    mc = getattr(model, "model_config", None)
    if getattr(mc, "type", None) == "system_one":
        from elbench.judges.system_one_judge import evaluate_basic_education_system_one

        return await evaluate_basic_education_system_one(
            api_base=mc.api_base,
            api_key=mc.api_key,
            model_name=str(mc.model or "liquid/d1-ultra"),
            timeout_seconds=float(CONFIG.globals.model_call_timeout_seconds or 60.0),
            retry_attempts=int(CONFIG.globals.retry.attempt or 1),
            system_prompt=system_prompt,
            messages=ops,
            format_fields=list(CONFIG.evaluation.format),
        )
    if CONFIG.evaluation.format_mode == "tool":
        tools = [generate_evaluation_tool()]

        # other_prompt = exported_result.replace_template(other_prompt)
        agent = create_react_agent(
            model=model.bind_tools(
                tools,
                tool_choice={
                    "type": "function",
                    "function": {"name": "save_result_to_database"},
                },  # type: ignore
                # tool_choice="required",
            ),
            tools=tools,
            # model=model,
            prompt=system_prompt
            + "\n\nPlease call the save_result_to_database tool to store the evaluation result.",
        )

        a = await asyncio.wait_for(
            agent.ainvoke({"messages": ops}),
            timeout=CONFIG.globals.model_call_timeout_seconds,
        )
        data = json.loads(a["messages"][-1].content)
        return data
    elif CONFIG.evaluation.format_mode == "prompt":
        format_example = CONFIG.evaluation.format_to_json_example()
        agent = create_react_agent(
            model=model,
            tools=[],
            prompt=system_prompt
            + "\n\nReturn exactly one JSON object wrapped with <START OF EVAL OUTPUT> and <END OF EVAL OUTPUT>."
            + "\nDo not add markdown fences, explanations, or extra text."
            + "\nUse the same keys and value types as this example:\n"
            + "\n<START OF EVAL OUTPUT>\n"
            + f"{format_example}\n"
            + "<END OF EVAL OUTPUT>",
        )

        a = await asyncio.wait_for(
            agent.ainvoke({"messages": ops}),
            timeout=CONFIG.globals.model_call_timeout_seconds,
        )
        response: str = a["messages"][-1].content

        # Fuck Gemini
        response = response.replace("\\<", "<")
        response = response.replace("\\>", ">")

        # print(response)

        # 姝ｅ垯琛ㄨ揪寮忓尮閰?START OUTPUT>鍜?END OUTPUT>
        match = re.findall(
            r"<START OF EVAL OUTPUT>(.*)<END OF EVAL OUTPUT>", response, re.DOTALL
        )
        if len(match) > 0:
            text = match[-1]
            # 濡傛灉text琚玚``鍖呭洿锛屽幓鎺?
            text = text.strip().strip("```").strip("json")
            # print(text, "#############")
            return json.loads(text)
        else:
            raise ValueError("Output does not contain <START OUTPUT> and <END OUTPUT>")
    else:
        raise ValueError(f"Invalid format mode: {CONFIG.evaluation.format_mode}")


if __name__ == "__main__":
    import asyncio

    async def main():
        from elbench.basic_education_runtime.model import init_chat_model_from_dict
        # from langchain.globals import set_debug

        # set_debug(True)

        ef = ExportFormat.from_json_file(
            "/tmp/basic_education_runtime/sample.json"
        )

        model = init_chat_model_from_dict(CONFIG.models["4o_teacher"])
        a = await evaluate(model, ef)
        print(a)

    asyncio.run(main())

