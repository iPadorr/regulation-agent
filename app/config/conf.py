"""
配置加载（++ OmegaConf + 环境变量解析思路）
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

# 获取配置文件
conf_path = Path(__file__).parent.parent.parent / "conf.yml"
conf = OmegaConf.load(conf_path)

# ++ 注册 env 解析器
OmegaConf.register_new_resolver("env", lambda name: os.getenv(name, ""))
OmegaConf.resolve(conf)


# ------------------------------------------------------------------
# ++ PostgreSQL（pgsql backend 时使用）
# ------------------------------------------------------------------
@dataclass
class PostgreSQLConfig:
    host: str = ""
    port: int = None
    user: str = ""
    password: str = ""
    database: str = ""


pgsql_config = PostgreSQLConfig(**conf.get("postgresql", {}))


# ------------------------------------------------------------------
# ++ 向量嵌入
# ------------------------------------------------------------------
@dataclass
class VectorStoreConfig:
    embedding_url: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""
    enable: bool = None


vector_store_config = VectorStoreConfig(**conf.get("vector_store", {}))


# ------------------------------------------------------------------
# 建筑规范文件的元信息
# ------------------------------------------------------------------
@dataclass
class RegulationConfig:
    code: str
    name: str
    file_path: str


regulation_config = RegulationConfig(**conf.regulation)


# ------------------------------------------------------------------
# RAG 后端与召回参数
# ------------------------------------------------------------------
@dataclass
class RagConfig:
    backend: str = "local"  # local | pgsql
    top_k: int = 5
    threshold: float = 0.2
    weight_vec: float = 0.6

rag_config = RagConfig(**conf.get("rag", {}))


# ------------------------------------------------------------------
# ++ LLM 模型池（LLMConfig LLMModelConfig）
# ------------------------------------------------------------------
@dataclass
class LLMModelConfig:
    # 基础参数
    base_url: str
    api_key: str
    model: str
    # 可接收其他额外参数（temperature,max_tokens)
    params: dict = field(default_factory=dict)


@dataclass
class LLMConfig:
    rag_model: str
    models: dict[str, LLMModelConfig]


_base_params = ["base_url", "api_key", "model"]

llm_config = LLMConfig(
    rag_model=conf.llm.rag_model,
    models={
        name: LLMModelConfig(
            **{k: v for k, v in data.items() if k in _base_params},
            params={k: v for k, v in data.items() if k not in _base_params},
        )
        for name, data in conf.llm.models.items()
    },
)


# ------------------------------------------------------------------
# 日志配置（复用原结构）
# ------------------------------------------------------------------
@dataclass
class LoggingConfig:
    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    log_dir: str = "logs"
    max_file_size: str = "10MB"


logging_config = LoggingConfig(**conf.get("logging", {}))
