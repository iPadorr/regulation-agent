"""
向量存储（条款级别）

++ app/core/vector_store.py 的 embed() 与 PostgreSQL 写入思路，
面向"规范条款"的向量入库与检索
"""
from loguru import logger  # 替代标准库logging
from openai import AsyncOpenAI  # AsyncOpenAI:openai 官方 SDK 的异步客户端，内部基于 httpx.AsyncClient，不会阻塞事件循环
from sqlalchemy import text  # 把原生 SQL 字符串包装成 SQLAlchemy 可执行对象。SQLAlchemy 2.0 以后，session.execute("SELECT * FROM t") 会报错
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession  # 异步数据库会话。配合 asyncpg 等异步驱动，在 async/await 代码里执行 CRUD。和 AsyncOpenAI 一样，保证数据库 IO 不阻塞主线程，能和 HTTP 请求、LLM 调用并发跑。

from app.config.conf import vector_store_config
from app.core.pdf_parser import Clause
from app.dto.reg_model import RegulationClause, EMBED_DIM


# 处理条款，生成嵌入文本向量
async def embed(texts: list[str]) -> list[list[float]]:
    """
    调用 OpenAI 兼容的 embedding API 将文本转向量。
    ++ app/core/vector_store.py:embed
    """
    if not texts:
        return []

    batch_size = 64  # 均衡值，以此处理64条规范条款
    try:
        embedding: list[list[float]] = []
        client = AsyncOpenAI(
            base_url=vector_store_config.embedding_url,
            api_key=vector_store_config.embedding_api_key,
        )
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            temp = await client.embeddings.create(
                model=vector_store_config.embedding_model,
                input=batch,
                encoding_format="float",
            )
            embedding.extend([item.embedding for item in temp.data])
        await client.close()
        return embedding
    except Exception as e:
        logger.exception(f"fail embed: {e}")
        return []


# 把生成的向量存入数据库
async def insert_clauses_vector(
        session: AsyncSession, clauses: list[Clause], embeddings: list[list[float]]
):
    """
    将条款向量化写入 PostgreSQL（pgvector）。
    唯一键：regulation_code + clause_num，冲突覆盖更新。
    """
    if len(clauses) != len(embeddings):
        raise ValueError("条款与向量数量不一致")

    # 把数据组装成一个个独立的字典（行）doc，对应embedding嵌入，一个条款一个字典
    rows = [
        {
            "regulation_code": c.regulation_code,  # 规范名
            "chapter_num": c.chapter_num,  # 章
            "chapter_title": c.chapter_title,
            "section_num": c.section_num, # 节
            "section_title": c.section_title,
            "clause_num": c.clause_num,  # 条款
            "text": c.text,  # 具体条款内容
            "sub_items": c.sub_items,
            "explanation": c.explanation,
            "embed": embeddings[i],
        }
        for i, c in enumerate(clauses)
    ]
    await session.execute(
        insert(RegulationClause)
        .values(rows)
        .on_conflict_do_update(
            index_elements=["regulation_code", "clause_num"],
            set_={
                "chapter_title": text("EXCLUDED.chapter_title"),
                "section_title": text("EXCLUDED.section_title"),
                "text": text("EXCLUDED.text"),
                "sub_items": text("EXCLUDED.sub_items"),
                "explanation": text("EXCLUDED.explanation"),
                "embed": text("EXCLUDED.embed"),
            },
        )
    )
    await session.commit()  # 落盘
    logger.info(f"saved {len(rows)} clause vectors")  # 打印日志


# 余弦相似度检索
async def search_clauses_vector(
        session: AsyncSession,
        query_embed: list[float],
        regulation_codes: list[str] | None = None,
        k: int = 5,
        threshold: float = 0.2,
) -> list[RegulationClause]:
    """
    ++ 余弦相似度向量检索（pgvector <=> 算子）。
    """

    # 定义过滤指定规范
    codes_filter = ""
    if regulation_codes:
        joined = ",".join(f"'{c}'" for c in regulation_codes)
        codes_filter = f"AND regulation_code IN ({joined})"

    # <=> （距离算子，计算两个向量相距多远）
    # :q::vector (:q 是占位符，PostgreSQL 的类型转换语法，意思是"把 :q 这个字符串当成向量类型来处理"）
    sql = text(f"""
        SELECT rc.*, r.name AS regulation_name,
               1 - (rc.embed <=> cast(:q as vector)) AS score
        FROM regulation_clause rc
        LEFT JOIN regulation r ON rc.regulation_code = r.code
        WHERE 1 - (rc.embed <=> cast(:q as vector)) > :threshold
        {codes_filter}
        ORDER BY rc.embed <=> cast(:q as vector)
        LIMIT :k
    """)
    rows = (
        await session.execute(
            sql,
            {
                "q": str(query_embed),
                "threshold": threshold,
                "k": k,
            },
        )
    ).fetchall()  # 把数据库返回的所有结果一次性拿回来，变成一个列表

    # 输出（打包成RegulationClause类)
    out: list[RegulationClause] = []
    for row in rows:
        c = RegulationClause()
        for key, value in row._mapping.items():
            if key == "embed" or key == "score":  # 跳过"向量"和"分数"这两列，不塞回对象
                continue
            try:
                setattr(c, key, value)  # （对象, "属性名", 属性值）
            except Exception:
                pass
        out.append(c)
    return out
