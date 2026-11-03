"""
建筑规范智能助手 FastAPI 应用入口

复用原项目 main.py 的结构：
    - lifespan：启动时初始化索引
    - CORS 中间件
    - trace 中间件
    - 静态文件挂载 + SPA fallback
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.config.conf import rag_config, regulation_config
from app.middleware.trace_middleware import gen_trace_req_id
from app.util.log import setup_logging

from dotenv import load_dotenv
import os

# 加载 .env 文件，将变量注入到系统环境变量中
load_dotenv()


# 根目录
BASE_DIR = Path(__file__).parent
# 前端页面目录
STATIC_DIR = BASE_DIR / "dist"


# 初始化本地索引
async def _init_local_index():
    """
    初始化本地索引：解析规范文件 -> 向量化 -> 落盘
    """
    from app.core import local_store
    from app.core.pdf_parser import parse_regulation
    from app.core.vector_store import embed as api_embed

    # 已有索引则直接加载，然后返回
    if local_store.clause_count() == 0:
        local_store.load_from_disk()
    if local_store.clause_count() > 0:
        return

    # 没有加载，则开始解析规范文件
    # 拼接成绝对路径
    file_path = regulation_config.file_path
    abs_path = (BASE_DIR / file_path).resolve() if not Path(file_path).is_absolute() else Path(file_path)
    # 如果找不到路径就报错
    if not abs_path.exists():
        print(f"[startup] regulation file not found: {abs_path}")
        return

    # 把规范拆分成chunks（条款具备天然的chunks性质）
    clauses = parse_regulation(
        str(abs_path),
        regulation_config.code,
        regulation_config.name,
    )

    # 嵌入（API 不可用时自动回退到本地哈希向量）
    texts = [c.text for c in clauses]
    vecs = await api_embed(texts) or local_store._hash_embed(texts)
    # 放入本地索引
    local_store.upsert_clauses(clauses, vecs)
    print(f"[startup] indexed {len(clauses)} clauses from {abs_path.name}")


async def _init_pg_index():
    """初始化 PostgreSQL 索引"""
    from app.core import db_service
    await db_service.load()


# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # 决定用哪个后端服务
    if rag_config.backend == "pgsql":
        await _init_pg_index()
    else:
        await _init_local_index()
    yield


app = FastAPI(lifespan=lifespan)  # 创建应用实例

# 增加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 全部放行
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 生成一个 trace_id（链路追踪号）  生成一个 request_id（请求号）
# 贴到 ContextVar 上
app.middleware("http")(gen_trace_req_id)


app.include_router(api_router)  # 注册api路由

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8090, reload=True)
