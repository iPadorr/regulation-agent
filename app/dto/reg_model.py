"""
建筑规范条款的 SQLAlchemy ORM 模型（PostgreSQL + pgvector 路径使用）

表结构：
    regulation           规范元信息
    regulation_clause    条款（含 pgvector 向量列 + tsvector 全文列）

备注：本地离线索引(local_store.py)不依赖这些表，仅当配置启用 PostgreSQL 时使用。
"""

# Vector 语义搜索，没有关键词仍能通过理解query找到相关条款
# TSVECTOR 用于命中 “关键词”
# 核心：同时存储两种索引，实现混合检索（Hybrid Search），结合两者的权重打分，获取最精准的召回结果

# Column 用于告诉 sqlalchemy 这个对应数据库表中的一列

# 兼顾：高性能搜索，灵活的数据存储（JSONB）

from sqlalchemy import Column, DateTime, Integer, String, Text, text as sa_text  # 常用列类型和工具
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR  # 二进制 JSON 列、全文搜索向量列
from sqlalchemy.orm import DeclarativeBase

# pgvector 为可选依赖，仅在 Postgres 路径下使用
try:
    from pgvector.sqlalchemy import Vector

    _HAS_PGVECTOR = True
except Exception:
    _HAS_PGVECTOR = False
    Vector = None

EMBED_DIM = 1024  # BAAI/bge-m3 输出维度、模型用于生成条款文本向量


class Base(DeclarativeBase):  # 作为声明式映射基类
    pass


class Regulation(Base):
    """规范元信息"""

    __tablename__ = "regulation"
    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(64), unique=True, nullable=False, comment="规范编号 GB 55037-2022")
    name = Column(String(128), nullable=False, comment="规范名称")
    file_path = Column(String(256), comment="源文件路径")
    clause_count = Column(Integer, default=0, comment="条款数量")
    created_at = Column(DateTime, server_default=sa_text("now()"))


class RegulationClause(Base):
    """规范条款（向量 + 全文双存）"""

    __tablename__ = "regulation_clause"
    id = Column(Integer, primary_key=True, autoincrement=True)
    regulation_code = Column(String(64), nullable=False, index=True, comment="所属规范编号")
    chapter_num = Column(String(16), comment="大章编号")
    chapter_title = Column(String(128), comment="大章标题")
    section_num = Column(String(16), comment="小节编号")
    section_title = Column(String(128), comment="小节标题")
    clause_num = Column(String(32), nullable=False, index=True, comment="条款编号 2.2.3")
    text = Column(Text, nullable=False, comment="条款正文（含子项）")  # Text 无长度限制
    sub_items = Column(JSONB, comment="子项列表")  # PostgreSQL 二进制 JSON，存结构化子项
    explanation = Column(Text, comment="条文说明")

    # 向量列与全文列：pgvector / tsvector 仅在 Postgres 路径下创建
    if _HAS_PGVECTOR:
        embed = Column(Vector(EMBED_DIM), comment="条款向量")
    text_tsv = Column(TSVECTOR, comment="条款全文向量")  # 仅仅依赖 PostgreSQL
    created_at = Column(DateTime, server_default=sa_text("now()"))
