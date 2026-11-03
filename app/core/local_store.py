"""
本地离线索引（无需 PostgreSQL / pgvector）

设计目标：
    在没有 PostgreSQL + pgvector 环境下，仍可完成 RAG 流程的本地端到端测试。

存储内容：
    1. 条款列表（含元数据 + 条文说明）  -> JSON
    2. 向量矩阵                       -> npy（numpy 2D 数组）
    3. 倒排索引                       -> JSON（jieba 分词 -> 命中 clause_id 列表）

检索路径：
    - 向量召回：numpy 实现余弦相似度
    - 全文召回：倒排索引 + TF-IDF 加权
    - 混合召回：两路 union，归一化加权后取 Top-K

对外提供与 vector_store / doc_store 等价的简单 API：
    async def embed(texts) -> list[list[float]]   （++ embed 逻辑或本地回退）
    def upsert_clauses(clauses, embeddings)
    def search(query, k) -> list[Clause]            （混合检索）
"""
import json
import math
import os
import re
import threading  # threading.Lock() 多线程锁，防止同时读写内存里的数据搞乱
from dataclasses import asdict  # 把 clause 转成字典，方便储存为 json
from pathlib import Path
from typing import Optional

import jieba
import numpy as np
from loguru import logger

from app.core.pdf_parser import Clause
from app.dto.reg_model import EMBED_DIM  # 导入嵌入向量维度

# ------------------------------------------------------------------
# 路径
# ------------------------------------------------------------------
_BASE_DIR = Path(__file__).parent.parent.parent / "data"  # 根目录的 data
_CLAUSES_FILE = _BASE_DIR / "clauses.json"
_VECTORS_FILE = _BASE_DIR / "vectors.npy"
_INVERTED_FILE = _BASE_DIR / "inverted_index.json"
_META_FILE = _BASE_DIR / "meta.json"

# ------------------------------------------------------------------
# 全局状态
# ------------------------------------------------------------------
_lock = threading.Lock()  # 防止内存被搞乱，fastAPI可能同时处理多个请求
_clauses: list[dict] = []  # [{clause}, {clause}, {clause}......] 条款列表，从内存读取
_vectors: Optional[np.ndarray] = None  # shape (N, EMBED_DIM)  # 向量矩阵 --> N 条条款 * 嵌入维度（向量）
_inverted: dict[str, list[int]] = {}  # 倒排索引，查关键词时直接命中 [word: [3, 24, 88]...]
_idf: dict[str, float] = {}  # 计算每个词的重要性分数
_embedded = False  # 标记是否已经加载好数据


# ------------------------------------------------------------------
# 嵌入：优先 OpenAI 兼容 API；不可用则用本地哈希伪向量
# 外部调用：await embed([i.text for i in clauses])
# ------------------------------------------------------------------
async def embed(texts: list[str]) -> list[list[float]]:
    """
    将文本嵌入转为向量。

    优先调用 OpenAI 兼容 API（++ BAAI/bge-m3）；
    若 API 未配置或调用失败，则回退为本地哈希向量（仅用于功能测试，语义质量远不及嵌入模型），保证 RAG 流程可端到端运行。
    """
    # 尝试 embed
    try:
        from app.config.conf import vector_store_config  # 导入向量库配置

        if vector_store_config.embedding_url and os.getenv(  # 判断api是否可用
                vector_store_config.embedding_api_key.lstrip("${env:").rstrip("}"),
                None,
        ):
            from app.core.vector_store import embed as _pg_embed  # 可用，则导入并使用嵌入和向量库
            vecs = await _pg_embed(texts)
            if vecs:
                return vecs
    except Exception as e:  # pragma: no cover
        logger.debug(f"local_store: 跳过 API 嵌入: {e}")

    # 本地回退：基于 jieba 分词的哈希向量（dim=EMBED_DIM）
    return _hash_embed(texts)


def _hash_embed(texts: list[str]) -> list[list[float]]:  # ["*****", "*****", "*****"......]
    """哈希嵌入（仅用于离线无 API 场景）"""
    out = []

    # 取条款
    for text in texts:
        # 创建条款的哈希向量
        vec = np.zeros(EMBED_DIM, dtype=np.float32)  # 创建全 0 数组，[0.0, 0.0, 0.0...1024个...]

        # 取词 （疏散、走道、净宽……）
        for word in jieba.cut(text):  # 对一个条款分词，并对切好的词进行哈希向量计算   这里jieba分词返回的是生成器
            if not word or word.isspace():  # 跳过空字符串、空格、换行、制表符
                continue
            h = abs(hash(word)) % EMBED_DIM  # python 哈希函数把词映射为整数，整数映射到 0~1023 之间
            vec[h] += 1.0  # 在同一次运行中哈希值一样，所以如果两个条款包含同样的词，那么h的位置都会+1，就能计算相似度
            # 双 hash 扩展特征分布
            h2 = (hash(word + "x") & 0xFFFFFFFF) % EMBED_DIM  # 计算第二个位置 h2
            vec[h2] += 0.5  # h2权重轻于h
        n = float(np.linalg.norm(vec))  # 计算欧几里德距离 sqrt(v[0]^2 + v[1]^2 + ...... + v[1023]^2)
        if n > 0:  # 避免除以 0
            vec = vec / n  # 归一化，保证比较的合理性
        out.append(vec.tolist())  # 转回普通列表，把条款向量按顺序添加到输出列表 out
    return out  # 把向量列表返回出去  [[vec],[vec],[vec],[vec]]


# ------------------------------------------------------------------
# 写入：upsert clauses
# ------------------------------------------------------------------
"""
变量结构备注
term_frequencies --> [{word: num, word: num...}, {word: num, word: num...}......] word 词，{}记录条款中的词频，len(word_frequencies) = len(clauses)
doc_frequencies --> {word:num, word:num.....}  num 在多少条款中出现过，面向整本规范文档
_inverted --> {word:[num, num...], word:[num, num...], word:[num, num...]}  num 出现在哪些条款
"""


def upsert_clauses(clauses: list[Clause], embeddings: list[list[float]] | None = None):
    """
    把条款写入本地
    把倒排索引 inverted_index 写入本地
    把元数据写入本地

    Args:
        clauses: 条款列表
        embeddings: 每条条款对应的向量；如果为 None 需异步生成（先 embed 再传入并调用）
    """
    global _clauses, _vectors, _inverted, _idf, _embedded

    if len(clauses) == 0:
        logger.warning("local_store: 空条款列表，跳过写入")
        return

    if embeddings is None or len(embeddings) != len(clauses):
        logger.warning("local_store: 向量缺失或数量不匹配，跳过向量索引")
        embeddings = [[0.0] * EMBED_DIM] * len(clauses)

    with _lock:
        _clauses = [asdict(clause) for clause in clauses]  # 转成普通字典 [{}, {}......]
        _vectors = np.asarray(embeddings, dtype=np.float32)  # 转numpy二维数组

        # 确保形状一直，避免崩溃。如果不一致退化为全 0 矩阵
        if _vectors.shape != (len(clauses), EMBED_DIM):
            _vectors = np.zeros((len(clauses), EMBED_DIM), dtype=np.float32)

        # 构建倒排索引
        # 清空旧的倒排索引和权重表，全量重建索引，非增量更新。
        _inverted.clear()
        _idf.clear()

        # 条款频率统计  统计词在多少个条款中出现过 --> 用于后续计算 IDF（逆条款频率）：IDF = log(总文条款 / 包含该词的文档数)。
        doc_frequencies: dict[str, int] = {}
        # 词频列表  一个条款内各词的出现次数 后续和 IDF 结合为 TF-IDF 权重，为后续的关键词检索和混合排序做准备
        term_frequencies: list[dict[str, int]] = []

        # 取条款对象（里面包含一个条款的全部属性）
        for c in clauses:
            # 我们搜索规范时，意图上可能根据章节、小节、条款、内容等进行搜索匹配，所以全部都包含进去，避免搜索遗漏
            text = (c.text or "") + " " + (c.chapter_title or "") + " " + (c.section_title or "") + " " + (
                        c.explanation or "")
            term_frequency: dict[str, int] = {}
            for w in jieba.cut(text):
                w = w.strip()
                if not w or w.isdigit() or len(w) <= 1 or re.match(r"^[^\w\u4e00-\u9fff]+$", w):  # 非字母数字下划线、及非中文
                    continue
                term_frequency[w] = term_frequency.get(w, 0) + 1  # 统计词频
            # 单条条款词频处理完，放入大词频列表中
            term_frequencies.append(term_frequency)

            # 添加整体文档词频
            for word in term_frequency:
                doc_frequencies[word] = doc_frequencies.get(word, 0) + 1

        N = len(clauses)
        for word, df in doc_frequencies.items():
            _idf[word] = math.log((N + 1) / (df + 1)) + 1.0  # 计算权重，重要性分数，使用导数是因为出现频率越低，重要性应该越高

        for i, tf in enumerate(term_frequencies):
            for word in tf:
                _inverted.setdefault(word, []).append(i)  # {"word":[0, 1, 6], "word2":[4, 55, 79]......}

        # 落盘保存，下次直接加载
        _BASE_DIR.mkdir(parents=True, exist_ok=True)
        with _CLAUSES_FILE.open("w", encoding="utf-8") as f:
            json.dump(_clauses, f, ensure_ascii=False)
        np.save(_VECTORS_FILE, _vectors)
        with _INVERTED_FILE.open("w", encoding="utf-8") as f:
            json.dump({"inverted": _inverted, "idf": _idf}, f, ensure_ascii=False)
        with _META_FILE.open("w", encoding="utf-8") as f:
            json.dump({"count": len(_clauses), "embed_dim": EMBED_DIM}, f, ensure_ascii=False)

        _embedded = True  # 标记完成哈希向量嵌入
        logger.info(f"local_store: 写入 {len(_clauses)} 条条款，{len(_inverted)} 个倒排词项")


def load_from_disk():
    """从磁盘加载索引到内存（进程重启后使用）"""
    global _clauses, _vectors, _inverted, _idf, _embedded
    with _lock:
        if _CLAUSES_FILE.exists():
            _clauses = json.loads(_CLAUSES_FILE.read_text(encoding="utf-8"))
        if _VECTORS_FILE.exists():
            _vectors = np.load(_VECTORS_FILE)
        if _INVERTED_FILE.exists():
            data = json.loads(_INVERTED_FILE.read_text(encoding="utf-8"))
            _inverted = {k: list(v) for k, v in data.get("inverted", {}).items()}
            _idf = data.get("idf", {})
        _embedded = bool(_clauses) and (_vectors is not None)
        logger.info(f"local_store: 从磁盘加载 {len(_clauses)} 条条款")


# ------------------------------------------------------------------
# 检索
# ------------------------------------------------------------------

# 向量检索（余弦相似度）
def _cos_search(query_vec: list[float], k: int, threshold: float) -> list[tuple[int, float]]:
    """
    向量余弦召回 -> [(clause_idx, score)]

    Args:
        query_vec: 用户提问向量
        k: 返回k条结果
        threshold: 相似度取回阈值。不要低于 threshold 分数的条款

    """
    if _vectors is None or _vectors.size == 0:
        return []
    q = np.asarray(query_vec, dtype=np.float32)  # 转成 numpy 数组，用来做矩阵运算
    q_L2 = float(np.linalg.norm(q, ord=2))  # 计算向量的 L2 范数
    if q_L2 == 0:
        return []
    # 归一化后做内积 = 余弦相似度k
    mat = _vectors / (np.linalg.norm(_vectors, axis=1, keepdims=True) + 1e-9)  # 归一化每一行的向量长度 --> (N, 1024)
    q_norm = q / q_L2  # 归一化 （1024, ) (一维，numpy 会当成列向量)
    sims = mat @ q_norm  # 求相似度 --> (N, 1024) @ 1024, ) = (N, )  点积余弦相似度（0～1）

    # argsort(-sims)：按相似度从高到低排序（默认从大到小，加负号变成从小到大），取前 k*3 个候选
    idxs = np.argsort(-sims)[:k * 3]  # 后面会做混合检索
    out = []
    for idx in idxs:
        s = float(sims[idx])
        if s >= threshold:
            out.append((int(idx), s))
    return out  # [(5, 0.34), (55, 0.78)......]


# 全文检索（倒排索引 + TF-IDF）关键词匹配
def _text_search(query: str, k: int) -> list[tuple[int, float]]:
    """倒排 + TF-IDF 召回"""
    if not _inverted:
        return []

    # 用户问题查询分词、统计词频
    q_words: dict[str, int] = {}  # 统计用户问题中，每个词的出现频率 (即 term_frequency)
    for w in jieba.cut(query):  # 分词
        w = w.strip()
        if not w or w.isdigit() or len(w) <= 1:
            continue
        q_words[w] = q_words.get(w, 0) + 1  # q_words -> dict(word:int, word:int...)   得到qtf

    # 召回候选
    candidates: set[int] = set()  # 速度快，不重复
    for word in q_words:
        if word in _inverted:
            candidates.update(_inverted[word])  # 添加候选条款  把所有包含问题中存在的关键词的条款一次放入作为候选

    # 如果找回不足（后补充）
    if len(candidates) < k:
        pass

    # 打分：词频+权重  TF-IDF
    scores: dict[int, float] = {}  # {条款1: 分数, 条款2: 分数...}
    for idx in candidates:
        if idx < 0 or idx >= len(_clauses):
            continue
        clause = _clauses[idx]  # 根据关键词从条款中取出具体条款
        text = (clause.get("text") or "") + " " + (clause.get("chapter_title") or "") + " " + (
                    clause.get("section_title") or "") + " " + (clause.get("explanation") or "")

        # 计算 query 各 word 在候选条款中的 tf
        score = 0.0
        for word, qtf in q_words.items():
            tf = text.count(word)  # 统计条款中，这个词的出现次数
            if tf == 0:
                continue
            idf = _idf.get(word, 1.0)
            # 条款词频、词重要性分数（稀有更有价值）、在提问中该词的频率（说的越多用户越在意）
            score += (1 + math.log(tf)) * idf * (1 + math.log(qtf))  # log限制高频词无限增长，相乘加到总分上
        if score > 0:
            scores[idx] = score  # 保存条款得分

    ranked = sorted(scores.items(), key=lambda x: -x[1])[: k * 3]  # 可能存在召回不足的情况（后续可增加做模糊匹配）
    return [(idx, score) for idx, score in ranked]


def search(query: str, query_vec: list[float] | None = None, k: int = 5,
           threshold: float = 0.2, weight_vec: float = 0.6) -> list[tuple[Clause, float]]:
    """
    混合检索（向量 + 全文）

    Args:
        query: 用户问题文本
        query_vec: 用户问题向量；None 时仅做全文检索
        k: 返回 Top-K
        threshold: 向量相似度阈值
        weight_vec: 向量分数权重（0~1，剩余给全文）

    Returns:
        [(Clause, score)]，按综合分降序
    """
    if not _embedded and not _clauses:
        load_from_disk()
    if not _clauses:
        return []

    # 向量检索和全文检索，混合检索，保证格式一致
    vec_hits = _cos_search(query_vec, k, threshold) if query_vec else []  # --> [(int, float), (int, float)...]
    text_hits = _text_search(query, k)  # --> [(int, float), (int, float)...]

    # 归一化
    def _norm(hits):
        if not hits:
            return {}
        max_score = max(score for _, score in hits)
        min_score = min(score for _, score in hits)
        rng = max(max_score - min_score, 1e-9)
        return {idx: (score - min_score) / rng for idx, score in hits}

    # 归一化后才能放在一起比较
    vec_norm = _norm(vec_hits)  # {idx: score, idx: score......}
    text_norm = _norm(text_hits)  # {idx: score, idx: score......}

    all_idx = set(vec_norm) | set(text_norm)  # set 对 key 做去重，并集

    # 根据权重混合得分
    combined: dict[int, float] = {}
    for idx in all_idx:
        # 同时命中后做权重计算
        combined[idx] = weight_vec * vec_norm.get(idx, 0.0) + (1 - weight_vec) * text_norm.get(idx, 0.0)

    ranked = sorted(combined.items(), key=lambda x: -x[1])[:k]  #  倒排，混合后直接取 k 个
    out: list[tuple[Clause, float]] = []
    for idx, score in ranked:
        clause = _clauses[idx]
        out.append((
            Clause(
                regulation_code=clause["regulation_code"],
                regulation_name=clause["regulation_name"],
                chapter_num=clause["chapter_num"],
                chapter_title=clause["chapter_title"],
                section_num=clause["section_num"],
                section_title=clause["section_title"],
                clause_num=clause["clause_num"],
                text=clause["text"],
                sub_items=clause.get("sub_items") or [],
                explanation=clause.get("explanation") or "",
            ),
            float(score),
        ))
    return out  # [(Clause, int), (Clause, int)...]


def clause_count() -> int:
    return len(_clauses)


if __name__ == "__main__":
    import asyncio
    import sys

    sys.path.insert(0, str(_BASE_DIR.parent))
    from app.core.pdf_parser import parse_regulation

    src = sys.argv[1] if len(sys.argv) > 1 else "regulations_src/建筑防火通用规范.docx"
    clauses = parse_regulation(src, "GB 55037-2022", "建筑防火通用规范")
    vecs = asyncio.run(embed([c.text for c in clauses]))
    upsert_clauses(clauses, vecs)
    hits = search("消防车道设置要求", vecs[0] if vecs else None, k=3)
    for c, s in hits:
        print(f"{s:.3f}  {c.clause_num}  {c.text[:60]}")
