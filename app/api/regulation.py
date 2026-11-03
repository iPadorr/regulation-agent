"""
建筑规范智能助手 API 路由
提供 “建筑规范智能助手” 的 HTTP API 接口，供前端或其他客户端调用
"""
import json
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter  # 路由
from fastapi.responses import StreamingResponse  # 用于向客户端返回流式数据
# BaseModel 用于定义请求体数据模型，实现自动校验和类型转换，Field 对模型字段添加额外约束和描述
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------------------->
from app.core.regulation_agent import rag_query, rag_stream
# from app.dto.reg_model import RegulationClause  # 导入 ORM 模型类，用面向对象的方式操作数据库

# 初始化执行，制造一个空的路由。后续所有装饰器都会将路由规则注册到这个实例上。最终这个实例会被包含到主应用的路由器中。
regulation_router = APIRouter()


class RagReq(BaseModel):  # 用于规范客户端 POST 请求体的格式
    """RAG 查询请求"""
    query: str = Field(..., description="用户问题")
    k: int = Field(5, description="召回条款数")


# 将复杂的 Python 对象递归转换为可 JSON 序列化的格式，用于在响应前统一处理
# def _serializer(obj: Any) -> Any:
#     if hasattr(obj, "model_dump"):  # Pydantic v2 的实例方法，用于将模型实例转为字典
#         return obj.model_dump()  # 有则直接调用 --返回--> dict
#     if hasattr(obj, "__dict__"):  # 检查对象是否有 __dict__ 属性（即是否为普通 Python 对象/数据类/自定义类等，且没有定义 __slots__）
#         return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}  # 过滤带下划线私有属性
#     if isinstance(obj, (list, tuple)):  # 判断是否为 列表 或 元组
#         return [_serializer(x) for x in obj]
#     if isinstance(obj, dict):
#         return {k: _serializer(v) for k, v in obj.items()}
#     return str(obj)


@regulation_router.get("/health")  # 把装饰器下面的函数注册到 /health
async def health():
    """健康检查"""
    return 200


@regulation_router.post("/query")
async def query(req: RagReq):
    """非流式：一次返回答案与引用"""
    result = await rag_query(req.query, k=req.k)
    return {
        "answer": result.answer,
        "citations": [asdict(c) for c in result.citations],
    }


@regulation_router.post("/chat")
async def chat(req: RagReq):
    """
    流式接口，沿用原项目协议：
        refs:<json>\n   召回到的条款列表（一次性，前端先渲染来源）
        <token>\n       流式 token（多次）
        end:<json>\n    结束，结构化结果 {answer, citations}
    """

    async def gen():
        async for chunk in rag_stream(req.query, k=req.k):
            yield chunk

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
