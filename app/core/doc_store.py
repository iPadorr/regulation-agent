"""
文档存储（条款级别全文检索）

++ app/core/doc_store.py tsvector 全文检索思路
改造为面向"规范条款"的全文入库与检索
"""
import re

import jieba
from loguru import logger
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.pdf_parser import Clause
from app.dto.reg_model import RegulationClause


# 切分文档
def _to_tsvector_format(text: str) -> str:
    """
    ++ 将文本分词为 PostgreSQL tsvector 格式。
    ++ 把中文切成 PostgreSQL 能搜的格式
    """
    # 输入text "疏散走道净宽不应小于1.1m"
    words = [
        w
        for w in jieba.cut(str(text).strip())  # 分词，去掉首尾的空格,['疏散', '走道', '净宽', '不应', '小于', '1.1', 'm']

        # w存在，去掉数字，非单字，非标点符号，['疏散', '走道', '净宽', '不应', '小于']
        if w and not w.isdigit() and len(w) > 1 and not re.match(r"^[^\w\u4e00-\u9fff]+$", w)
    ]
    if not words:
        return ""
    # 用空格拼接字符串，最后输出 "散 走道 净宽 不应 小于"，由于SQL和SHELL的语法，英文的空格和单引号会被特殊处理
    return " ".join(
        f"'{w.replace("'", "''")}'" if " " in w or "'" in w else w
        for w in words
    )


# 把切分好的文档入库（增加子项，先text精确定位，后结构化展示和精确引用）
async def insert_clauses_doc(session: AsyncSession, clauses: list[Clause]):
    """
    写入条款全文索引（tsvector 列）。
    ++ insert_doc_store
    """
    rows = [
        {
            "regulation_code": c.regulation_code,  # 规范明
            "chapter_num": c.chapter_num,
            "chapter_title": c.chapter_title,  # 章
            "section_num": c.section_num,
            "section_title": c.section_title,  # 节
            "clause_num": c.clause_num,
            "text": c.text,  # 具体条款内容
            "sub_items": c.sub_items,
            "explanation": c.explanation,  # 补充说明
            "text_tsv": _to_tsvector_format(
                " ".join(
                    filter(
                        None,
                        [
                            c.chapter_title,
                            c.section_title,
                            c.clause_num,
                            c.text,
                            c.explanation,
                        ],
                    )
                )
            ),
        }
        for c in clauses
    ]
    await session.execute(
        insert(RegulationClause)
        .values(rows)
        .on_conflict_do_update(
            index_elements=["regulation_code", "clause_num"],
            set_={
                "text_tsv": text("EXCLUDED.text_tsv"),
                "text": text("EXCLUDED.text"),
                "sub_items": text("EXCLUDED.sub_items"),
                "explanation": text("EXCLUDED.explanation"),
            },
        )
    )
    await session.commit()
    logger.info(f"saved {len(rows)} clause docs")


# 文档检索
async def search_clauses_doc(
        session: AsyncSession,
        query: str,
        regulation_codes: list[str] | None = None,
        k: int = 5,
) -> list[RegulationClause]:
    """
    ++ 全文检索
    """
    codes_filter = ""

    # 切分query
    query_terms = _to_tsvector_format(query)
    if not query_terms:
        return []

    params: dict = {"query": query, "k": k}

    # 如果指定了只搜某几本规范（比如只搜 GB 55037），就加上过滤条件。
    if regulation_codes:
        codes_filter = "AND regulation_code = ANY(:codes)"  # :codes 是占位符，ANY(:codes) 列表里的任意一个
        params["codes"] = regulation_codes

    # 把用户的查询词转成搜索格式。'simple' 表示不做额外的语言处理，直接按空格分词匹配.@@ 匹配运算符，to_tsquery 和 jieba分词才能匹配对上
    sql = text(f"""
        SELECT rc.*, r.name AS regulation_name
        FROM regulation_clause rc
        LEFT JOIN regulation r ON rc.regulation_code = r.code
        WHERE (rc.text_tsv @@ plainto_tsquery('simple', :query))
        {codes_filter}
        LIMIT :k
    """)
    rows = (await session.execute(sql, params)).fetchall()
    out: list[RegulationClause] = []
    for row in rows:
        c = RegulationClause()
        for key, value in row._mapping.items():
            if key == "text_tsv":
                continue
            try:
                setattr(c, key, value)
            except Exception:
                pass
        out.append(c)
    return out
