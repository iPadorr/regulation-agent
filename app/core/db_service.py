"""
数据库服务（PostgreSQL + pgvector 路径）

++ app/core/db_service.py 会话工厂与生命周期管理思路：
    - _create_engine / _get_sessionmaker / pg_session / cleanup 复用
    - load() 解析规范文件 -> 向量化 -> 写入 PG

配置启用 PostgreSQL 时被调用；离线模式下走 local_store。
PostgreSQL 数据库的整体管理服务：连数据库、建表、储存规范条款、检索
"""
import asyncio
from contextlib import asynccontextmanager  # 自动打开关闭连接
from typing import AsyncGenerator  # 类型提示
from pathlib import Path

from loguru import logger
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import (
    AsyncEngine,  # 连接池类型
    AsyncSession,  # 会话
    async_sessionmaker,  # 会话工厂
    create_async_engine,  # 数据库连接池
)

from app.config.conf import pgsql_config, regulation_config, vector_store_config
from app.core.doc_store import insert_clauses_doc, search_clauses_doc
from app.core.vector_store import embed, insert_clauses_vector, search_clauses_vector
from app.core.pdf_parser import parse_regulation
from app.dto.reg_model import Regulation, RegulationClause

# 拼接数据库地址
# ++ PostgreSQL URL 构造方式 ---拼接成类似网址---> postgresql+asyncpg://regulation:123321@127.0.0.1:5432/regulation
pgsql_url = (
    f"postgresql+asyncpg://{pgsql_config.user}:{pgsql_config.password}"  # 用PostgreSQL，驱动选 asyncpg（异步）账号密码
    f"@{pgsql_config.host}:{pgsql_config.port}/{pgsql_config.database}"  # 地址 端口 数据库名
)

# 定义全局单例：数据库连接池（要建网络连接、协商协议）
# 池很重：只应造一次，然后大家共用。先占位，第一次用时真正创建
_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker | None = None


# 创建连接池
def _create_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            pgsql_url,
            pool_size=10,  # 常驻10个连接
            max_overflow=20,  # 忙的时候可增加20个，即最终达到30个
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )
    return _engine


# 造会话工厂
def _get_sessionmaker() -> async_sessionmaker:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            _create_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _sessionmaker


# 上下文管理器
@asynccontextmanager
async def pg_session() -> AsyncGenerator[AsyncSession, None]:
    async with _get_sessionmaker()() as session:  # 异步打开，启动关闭。拿到工厂，产生一个session
        yield session


# 程序关闭时调用，释放所有数据库连接，防止内存泄漏
async def cleanup():
    global _engine, _sessionmaker
    if _engine is not None:
        try:
            await _engine.dispose()
        except RuntimeError:
            pass
    _engine = None
    _sessionmaker = None


# ------------------------------------------------------------------
# 建表（DDL）：仅用于初始化 pgvector 扩展与表结构
# ------------------------------------------------------------------

_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS regulation (
    id SERIAL PRIMARY KEY,
    code VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(128) NOT NULL,
    file_path VARCHAR(256),
    clause_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS regulation_clause (
    id SERIAL PRIMARY KEY,
    regulation_code VARCHAR(64) NOT NULL,
    chapter_num VARCHAR(16),
    chapter_title VARCHAR(128),
    section_num VARCHAR(16),
    section_title VARCHAR(128),
    clause_num VARCHAR(32) NOT NULL,
    text TEXT NOT NULL,
    sub_items JSONB,
    explanation TEXT,
    embed VECTOR(1024),
    text_tsv TSVECTOR
);
CREATE INDEX IF NOT EXISTS idx_rc_code ON regulation_clause(regulation_code);
CREATE INDEX IF NOT EXISTS idx_rc_clause ON regulation_clause(clause_num);
CREATE UNIQUE INDEX IF NOT EXISTS uq_rc_code_clause
    ON regulation_clause(regulation_code, clause_num);
CREATE INDEX IF NOT EXISTS idx_rc_tsv ON regulation_clause USING GIN(text_tsv);
"""

# 解释
# explaintion = """
# CREATE EXTENSION IF NOT EXISTS vector;  -- 启用 pgvector 扩展，不然没法存向量
#
# CREATE TABLE IF NOT EXISTS regulation (
#     id SERIAL PRIMARY KEY,
#     code VARCHAR(64) UNIQUE NOT NULL,    -- 规范编号，如 GB 55037-2022
#     name VARCHAR(128) NOT NULL,          -- 规范名称
#     file_path VARCHAR(256),              -- 源文件路径
#     clause_count INTEGER DEFAULT 0,      -- 条款数量
#     created_at TIMESTAMP DEFAULT now()   -- 创建时间
# );
#
# CREATE TABLE IF NOT EXISTS regulation_clause (
#     id SERIAL PRIMARY KEY,
#     regulation_code VARCHAR(64) NOT NULL,  -- 所属规范
#     chapter_num VARCHAR(16),               -- 章号
#     chapter_title VARCHAR(128),            -- 章标题
#     section_num VARCHAR(16),               -- 节号
#     section_title VARCHAR(128),            -- 节标题
#     clause_num VARCHAR(32) NOT NULL,       -- 条款号，如 2.2.3
#     text TEXT NOT NULL,                    -- 条款正文
#     sub_items JSONB,                       -- 子项，JSONB 格式
#     explanation TEXT,                      -- 条文说明
#     embed VECTOR(1024),                   -- 向量（1024 维，bge-m3）
#     text_tsv TSVECTOR                     -- 全文检索索引
# );
#
# -- 索引：加快查询速度
# CREATE INDEX IF NOT EXISTS idx_rc_code ON regulation_clause(regulation_code);
# CREATE INDEX IF NOT EXISTS idx_rc_clause ON regulation_clause(clause_num);
# CREATE UNIQUE INDEX IF NOT EXISTS uq_rc_code_clause
#     ON regulation_clause(regulation_code, clause_num);  -- 唯一：规范+条款号不能重复
# CREATE INDEX IF NOT EXISTS idx_rc_tsv ON regulation_clause USING GIN(text_tsv);  -- 全文检索加速
# """



# 异步执行建表，拆分，逐行执行
async def init_schema():
    """初始化 PG schema"""
    async with pg_session() as session:
        from sqlalchemy import text as sa_text

        for stmt in _DDL.split(";"):  # 按分号拆开，一条条执行
            if stmt.strip():  # 跳过空字符串
                await session.execute(sa_text(stmt))
        await session.commit()  # 落盘
    logger.info("PG schema initialized")  # 打印日志


# ------------------------------------------------------------------
# 加载：解析规范文件 -> 入库
# ------------------------------------------------------------------
async def load():
    """解析配置中的规范文件，向量化后写入 PostgreSQL"""
    await init_schema()  # 确保表存在

    # 检查文件
    src_path = regulation_config.file_path
    if not src_path or not Path(src_path).exists():
        logger.error(f"regulation file not found: {src_path}")
        return

    logger.info(f"parsing regulation: {src_path}")
    # 把 docx 文件解析成一堆 Clause 对象（章、节、条款号、正文等）
    clauses = parse_regulation(
        src_path,
        regulation_config.code,
        regulation_config.name,
    )
    logger.info(f"parsed {len(clauses)} clauses")

    # 写入规范元信息
    async with pg_session() as session:
        from sqlalchemy import text as sa_text

        await session.execute(
            insert(Regulation)
            .values(
                code=regulation_config.code,
                name=regulation_config.name,
                file_path=src_path,
                clause_count=len(clauses),
            )
            .on_conflict_do_update(
                index_elements=["code"],
                set_={
                    "name": sa_text("EXCLUDED.name"),
                    "file_path": sa_text("EXCLUDED.file_path"),
                    "clause_count": sa_text("EXCLUDED.clause_count"),
                },
            )
        )
        await session.commit()

    # 向量化
    if vector_store_config.enable:
        texts = [c.text for c in clauses]
        embeddings = await embed(texts)
        if len(embeddings) != len(clauses):
            logger.error("embedding count mismatch, abort vector insert")
            return
        async with pg_session() as session:
            await insert_clauses_vector(session, clauses, embeddings)
            await insert_clauses_doc(session, clauses)
    else:
        # 未启用向量，仍写入条款文本（无 embed 列）
        async with pg_session() as session:
            await insert_clauses_doc(session, clauses)


# ------------------------------------------------------------------
# 检索：向量 + 全文
# ------------------------------------------------------------------
async def vector_recall(query_embed: list[float], k: int = 5, threshold: float = 0.2):
    async with pg_session() as session:
        return await search_clauses_vector(
            session, query_embed, regulation_codes=None, k=k, threshold=threshold
        )


async def doc_recall(query: str, k: int = 5):
    async with pg_session() as session:
        return await search_clauses_doc(
            session, query, regulation_codes=None, k=k
        )


if __name__ == "__main__":
    asyncio.run(load())
