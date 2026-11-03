"""
建筑规范智能助手 RAG Agent

简述：
    -> 拿到用户问题
    -> 分析并检索相关条款
    -> 把条款塞给大模型
    -> 结合用户问题，让大模型组织语言回答
    -> 把回答和引用一起返回给前端。

流程：
    1. 检索：向量 + 全文混合召回 Top-K 条款
    2. 生成：将条款作为引用上下文，调用 LLM 输出正文 + 引用
    3. 输出：流式产出 <回答> 与 <引用> 两段，前端可分别渲染

存储后端选择：
    - 默认：app.core.local_store（本地离线索引，无需 Postgres）
    - 可选：app.core.db_service（PostgreSQL + pgvector）
      通过 conf.yml 中 rag.backend = pgsql 切换
"""
import asyncio
import json
import re
from dataclasses import asdict, dataclass
from typing import AsyncGenerator, Literal

from loguru import logger

# ---------------------------------------------------------------------------------------->
import app.util
from app.config.conf import llm_config, rag_config
from app.core.pdf_parser import Clause
from app.util import ask_llm, get_prompt, parse_json

_ANSWER_TAG_RE = re.compile(r"</?(回答|引用)>")


# ------------------------------------------------------------------
# DTO
# ------------------------------------------------------------------
@dataclass
class Citation:
    """单条引用"""

    # 规范
    regulation_code: str
    regulation_name: str
    # 章
    chapter_num: str
    chapter_title: str
    # 节
    section_num: str
    section_title: str
    # 条款
    clause_num: str
    # 具体内容
    text: str
    explanation: str = ""

    def citation_str(self) -> str:
        """
        生成刻度的引用字符串
        返回 -> 《建筑防火通用规范》 (GB-22222) 第2章 基本规定 第2.2节 消防通道 第2.2.2条
        """

        parts = [f"《{self.regulation_name}》({self.regulation_code})"]
        if self.chapter_title:
            parts.append(f"第{self.chapter_num}章 {self.chapter_title}")
        if self.section_title and self.section_num != self.clause_num:
            parts.append(f"第{self.section_num}节 {self.section_title}")
        parts.append(f"第{self.clause_num}条")
        return " ".join(parts)


@dataclass
class RagAnswer:
    """RAG 输出"""
    answer: str
    citations: list[Citation]


# ------------------------------------------------------------------
# 检索后端抽象（根据配置匹配检索仓库）
# ------------------------------------------------------------------
def _row_to_clause(row) -> Clause:
    return Clause(
        regulation_code=row.regulation_code,
        regulation_name=getattr(row, "regulation_name", ""),
        chapter_num=row.chapter_num or "",
        chapter_title=row.chapter_title or "",
        section_num=row.section_num or "",
        section_title=row.section_title or "",
        clause_num=row.clause_num,
        text=row.text,
        sub_items=getattr(row, "sub_items", []) or [],
        explanation=getattr(row, "explanation", "") or "",
    )


async def _recall_local(query: str, query_vec: list[float] | None, k: int) -> list[
    tuple[Clause, float]]:  # [(Clause, score),(Clause, score)...]
    """本地离线索引检索"""
    from app.core import local_store
    if not local_store.clause_count():  # 判断是否加载了本地库
        local_store.load_from_disk()
    return local_store.search(query, query_vec, k=k)


async def _recall_pg(query: str, k: int) -> list[tuple[Clause, float]]:  # [(Clause, score),(Clause, score)...]
    """PostgreSQL + pgvector 检索"""
    from app.core import db_service
    from app.core.vector_store import embed as api_embed

    q_vec = (await api_embed([query]) or [None])[0]  # 获取嵌入向量 -> [float, ...] or None

    # API 嵌入不可用时，降级到本地索引（哈希向量 + 倒排），保证离线召回效果与 local 模式一致
    if q_vec is None:
        from app.core import local_store
        if not local_store.clause_count():
            local_store.load_from_disk()
        hash_vec = local_store._hash_embed([query])[0]
        return local_store.search(query, hash_vec, k=k)

    # 同时启动两个异步任务
    vec_hits, doc_hits = await asyncio.gather(
        # 向量检索，召回条款
        db_service.vector_recall(q_vec, k=k),
        # 全文检索
        db_service.doc_recall(query, k=k),
    )

    # 合并去重
    check_hits = set()
    out: list[tuple[Clause, float]] = []  # -> [(Clause, score), ...]
    for hit in vec_hits:
        key = (hit.regulation_code, hit.clause_num)
        if key in check_hits:
            continue
        check_hits.add(key)
        out.append((_row_to_clause(hit), 0.9))  # 向量检索更可信
    for hit in doc_hits:
        key = (hit.regulation_code, hit.clause_num)
        if key in check_hits:
            continue
        check_hits.add(key)
        out.append((_row_to_clause(hit), 0.5))
    return out


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

# LLM 各种挂了的情况，使用 fallback_answer
def _fallback_answer(query: str, hits: list[tuple[Clause, float]]) -> str:
    """
    无 LLM 时的兜底输出：把检索到的条款直接拼装为 <回答> + <引用> 格式，
    保证 RAG 流程在无 API Key 的测试环境下也能给出有规范来源的引用式回答。
    """

    # 如果没有检索到任何条款，返回一个保护用户体验的提示
    if not hits:
        return (
            "<回答>未在已加载规范中找到与该问题直接相关的条款。</回答>\n"
            "<引用></引用>"
        )
    parts = ["<回答>"]
    parts.append(f"针对「{query}」，从《{hits[0][0].regulation_name}》中检索到以下相关条款：")
    for i, (c, _) in enumerate(hits, 1):
        # 控制内容长度
        text = c.text.strip()
        if len(text) > 300:  # 截断超长 text，一般情况都能覆盖
            text = text[:300] + "……"
        parts.append(f"{i}. {c.citation()}\n   {text}")  # 序号 引述 条款内容
    parts.append("（说明：当前为离线模式，未调用 LLM 生成综合答案；以上为检索到的规范条款原文，可作为引用依据使用。）")
    parts.append("</回答>")
    parts.append("<引用>")
    for i, (c, _) in enumerate(hits, 1):
        parts.append(f"{i}. {c.citation()} - {c.text.strip()[:80]}")
    parts.append("</引用>")
    return "\n".join(parts)


# 给 LLM 模型格式化参考条款，便于大语言模型分析
def _format_references(hits: list[tuple[Clause, float]]) -> str:
    """将召回条款格式化为给 LLM 的引用块"""
    if not hits:
        return "（无相关规范条款）"
    lines = []
    for i, (c, _) in enumerate(hits, 1):
        meta = c.citation()  # 条款引述
        text = c.text.strip()  # 条款内容
        # 截断超长条款以控制上下文长度
        if len(text) > 800:  # 如有例外：加快速度，节省token （一般规范各个条款不会出现超过800）
            text = text[:800] + "……"
        expl = ""
        if c.explanation:
            e = c.explanation.strip()
            # 控制内容长度
            if len(e) > 300:
                e = e[:300] + "……"
            expl = f"\n   条文说明：{e}"  # 条款说明
        lines.append(f"[{i}] {meta}\n   正文：{text}{expl}")
    return "\n\n".join(lines)  # \n\n可让 LLM 看的更清晰，类似于1、2  [1] [2]


# 非流式
async def rag_query(query: str, k: int = 5) -> RagAnswer:
    """
    非流式：一次返回完整答案与引用。
    """
    hits = await _recall(query, k)  # 根据服务状态，选择哈希索引或pg索引
    references = _format_references(hits)  # 给 LLM 的条款参考
    # 渲染提示词模板
    prompt = get_prompt(
        prompt_file="regulation_rag",
        prompt_name="regulation_rag",
        query=query,
        references=references,
    )
    # 组织 OpenAI 格式的消息列表
    messages = [
        {"role": "system", "content": prompt["system"]},  # 告诉系统【身份】，【要求】，以及【可引用内容】
        {"role": "user", "content": prompt["user"]},
    ]

    # 优先调用真实 LLM，失败时回退到 “直接拼接检索条款” 的模式
    raw = ""
    max_retries = 3
    for i in range(max_retries):
        try:
            raw = await ask_llm(llm_config.rag_model, messages, timeout=120.0)
            break  # LLM 成功调用，退出循环
        except Exception as e:
            logger.warning(f"LLM call failed, fallback to citation-only mode: {e}")
            if i == max_retries - 1:
                try:
                    # raw = _fallback_answer(query, hits)  # 调用 LLM 失败，退回
                    raw = "AI 回答暂时不可用，已根据您的问题查询到以下规范条文，请参考下方溯源引用。"
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raw = "服务器暂时无法生成答案，请稍后重试"
    answer_text, citations = _parse_answer(raw, hits)  # 用正则，从 LLM 输出（或兜底方案输出）提出<回答>和<引用>
    return RagAnswer(answer=answer_text, citations=citations)  # 包装成 RagAnswer 对象返回


# 异步生成器，每次 yield 出一个字符串
async def rag_stream(query: str, k: int = 5) -> AsyncGenerator[str, None]:
    """
    流式输出，遵循协议：
        <token>\n           # 流式 token
        end:<json>\n        # 结束时一次性返回结构化结果
    """
    try:
        hits = await _recall(query, k)  # 召回命中的条款，自动选择 PG 或 Hash
        references = _format_references(hits)  # 条款对象列表 -> 格式化为引用字符串
        prompt = get_prompt(
            "regulation_rag",
            "regulation_rag",
            query=query,
            references=references,
        )
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]

        # 先把检索到的引用源列表推送一次（前端可立即展示来源）
        # 保证用户体验，用户感受 -> 立马给出引用源（响应迅速），有引用源（可靠）
        refs_payload = [
            {
                "regulation_code": c.regulation_code,
                "regulation_name": c.regulation_name,
                "chapter_num": c.chapter_num,
                "chapter_title": c.chapter_title,
                "section_num": c.section_num,
                "section_title": c.section_title,
                "clause_num": c.clause_num,
                "text": c.text,
                "explanation": c.explanation,
            }
            for c, _ in hits  # c -> Clause
        ]

        # 列表转为JSON字符串，中文字符不转义为 ascii 码
        yield f"refs:{json.dumps(refs_payload, ensure_ascii=False)}\n"  # 抛出引用内容字符串  refs:[{"":"", "":"", ...}, {}...]

        # 调用 LLM 流式接口（沿用 ask_llm 的同步调用 + 分块输出）
        # 若 LLM 不支持流式 或者 无 API Key，则回退为本地拼装并一次性推送

        full_text = ""  # 用于存放 LLM 返回的所有文字
        used_fallback = False
        try:
            from openai import AsyncOpenAI  # 创建异步客户端
            from app.config.conf import llm_config as _cfg

            # 获取 RAG 模型名
            model_cfg = _cfg.models[_cfg.rag_model]
            client = AsyncOpenAI(base_url=model_cfg.base_url, api_key=model_cfg.api_key)  # 配置信息
            stream = await client.chat.completions.create(
                model=model_cfg.model,  # 模型
                messages=messages,
                stream=True,
                **model_cfg.params,  # 额外参数：(如 temperature、max_tokens 等等)
            )

            # 流式接收（每次取一个数据块）
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta_temp = chunk.choices[0].delta  # OpenAI的流式格式，变化量对象
                content = getattr(delta_temp, "content", None)  # 安全取 content 的字段的内容，如果没有返回 None
                if content:
                    content = content.replace("<回答>", "").replace("</回答>", "")
                    full_text += content
                    yield f"{content}\n"  # 抛出给前端
            await client.close()  # 流结束，关闭 HTTP 客户端
        except Exception as e:
            # # 回退到本地拼装
            # logger.warning(f"stream LLM failed, fallback to citation-only mode: {e}")
            # full_text = _fallback_answer(query, hits)  # 调用回退回答，LLM 无法生成综合答案，用索引到的原文直接生成答案
            # used_fallback = True
            # # 分块输出，模拟流式
            # lines = full_text.splitlines(keepends=True)  # 保留换行符
            # for line in lines:
            #     yield line
            #     await asyncio.sleep(0.01)  # 避免过快，保障体验
            logger.warning(f"stream LLM failed, fallback to citation-only mode: {e}")
            full_text = "AI 回答暂时不可用，已根据您的问题查询到以下规范条文，请参考下方溯源引用。"
            used_fallback = True
            yield f"{full_text}\n"
        if used_fallback:
            # 兜底路径直接产出，需要逐行 yield 已完成
            # 后续完善额外处理
            pass

        # 从完整文本里提取（真流式，假流式 结束后，统一解析并发送end
        answer_text, citations = _parse_answer(full_text, hits)  # 用正则，从 LLM 输出（或兜底方案输出）提出<回答>和<引用>
        end_payload = {
            "answer": answer_text,
            "citations": [asdict(c) for c in citations],
        }
        yield f"end:{json.dumps(end_payload, ensure_ascii=False)}\n"  # 加 end 结束标记，前端接收后解析最终结构化数据
    except Exception as e:
        import traceback
        err = json.dumps(
            {"error": str(e), "details": traceback.format_exc()},
            ensure_ascii=False,
        )
        yield f"end:{err}\n"


async def _recall(query: str, k: int) -> list[tuple[Clause, float]]:
    """根据配置选择检索后端"""
    backend = rag_config.backend
    if backend == "pgsql":
        return await _recall_pg(query, k)  # 如果有 pg 能用，使用 pg

    # 默认本地
    from app.core import local_store
    q_vec = None

    # 优先用 API 嵌入；不可用则用本地哈希
    try:
        from app.core.vector_store import embed as api_embed
        vecs = await api_embed([query])
        if vecs:
            q_vec = vecs[0]
    except Exception as e:
        logger.debug(f"API embed unavailable, use local hash: {e}")

    # 如果 api 不能用，退回本地哈希向量
    if q_vec is None:
        q_vec = local_store._hash_embed([query])[0]
    return await _recall_local(query, q_vec, k)


# ------------------------------------------------------------------
# 格式化输出
# ------------------------------------------------------------------
_RE_ANSWER = re.compile(r"<回答>(.*?)</回答>", re.DOTALL)
_RE_REFERENCES = re.compile(r"<引用>(.*?)</引用>", re.DOTALL)


def _parse_answer(raw: str, hits: list[tuple[Clause, float]]) -> tuple[str, list[Citation]]:
    """从 LLM 输出中抽取 <回答> 与 <引用>"""
    # 抽取 <回答>
    answer_match = _RE_ANSWER.search(raw)
    answer_text = answer_match.group(1).strip() if answer_match else raw.strip()

    # 抽取 < 引用 >
    # 引用块：让 LLM 自行列出，但只接受出现在检索结果里的条款
    references_match = _RE_REFERENCES.search(raw)
    references: list[Citation] = []
    if references_match:
        block = references_match.group(1)
        # 匹配 "x.x.x" 的条款编号
        for m in re.finditer(r"\b(\d+\.\d+\.\d+[A-Z]?)\b", block):  # 在引用块里面找到所有 x.x.x 格式的条款编号
            clause_num = m.group(1)

            # 匹配条款号、去重、创建 Citation 对象放入列表
            for c, _ in hits:
                if c.clause_num == clause_num and not any(
                        x.clause_num == clause_num for x in references
                ):
                    references.append(
                        Citation(
                            regulation_code=c.regulation_code,
                            regulation_name=c.regulation_name,
                            chapter_num=c.chapter_num,
                            chapter_title=c.chapter_title,
                            section_num=c.section_num,
                            section_title=c.section_title,
                            clause_num=c.clause_num,
                            text=c.text,
                            explanation=c.explanation,
                        )
                    )
                    break

    # 兜底：若 LLM 没输出引用块，但检索到了条款，则全部作为引用返回
    if not references and hits:
        for c, _ in hits:
            references.append(
                Citation(
                    regulation_code=c.regulation_code,
                    regulation_name=c.regulation_name,
                    chapter_num=c.chapter_num,
                    chapter_title=c.chapter_title,
                    section_num=c.section_num,
                    section_title=c.section_title,
                    clause_num=c.clause_num,
                    text=c.text,
                    explanation=c.explanation,
                )
            )
    return answer_text, references


if __name__ == "__main__":
    # 命令行快速测试
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "消防车道设置要求"
    result = asyncio.run(rag_query(q))
    print("=== 回答 ===")
    print(result.answer)
    print("\n=== 引用 ===")
    for c in result.citations:
        print(c.citation_str())
