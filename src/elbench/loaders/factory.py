from __future__ import annotations

from elbench.loaders.base import BaseLoader
from elbench.loaders.basic_education_yaml_loader import BasicEducationYamlLoader
from elbench.loaders.jsonl_loader import JsonlLoader
from elbench.loaders.general_capability_loader import (
    AIMEJsonlLoader,
    CEvalJsonlLoader,
    IFEvalJsonlLoader,
    MATH500JsonlLoader,
    MMLUProJsonlLoader,
)
from elbench.loaders.xlsx_loader import XlsxLoader


class LoaderFactory:
    _LOADERS: dict[str, type[BaseLoader]] = {
        "basic_education_yaml": BasicEducationYamlLoader,
        "jsonl": JsonlLoader,
        "aime_jsonl": AIMEJsonlLoader,
        "ceval_jsonl": CEvalJsonlLoader,
        "ifeval_jsonl": IFEvalJsonlLoader,
        "math_500_jsonl": MATH500JsonlLoader,
        "mmlu_pro_jsonl": MMLUProJsonlLoader,
        "xlsx": XlsxLoader,
    }

    @classmethod
    def create(cls, loader_name: str) -> BaseLoader:
        if loader_name not in cls._LOADERS:
            raise ValueError(f"Unsupported loader: {loader_name}")
        return cls._LOADERS[loader_name]()
