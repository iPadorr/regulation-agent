"""
建筑规范文件解析器

将建筑规范 PDF / DOCX 文件解析为结构化的条款列表：
    规范编号 / 规范名称 / 大章(章节) / 小节 / 条款编号 / 条款正文 / 子项 / 条文说明

解析策略：
    - DOCX：直接解压 word/document.xml，按段落抽取文本（无第三方依赖、结构最清晰）
    - PDF ：使用 pypdf 按页抽取文本，再按换行切分为段落（作为兜底）

条款编号采用国标通用的三级结构：<章>.<节>.<条>，例如 2.2.3
"""
import os
import re
import zipfile
from dataclasses import dataclass, field


# ------------------------------------------------------------------
# 数据结构（包含单条条款所有信息）
# ------------------------------------------------------------------
@dataclass
class Clause:
    """单条规范条款"""
    regulation_code: str  # 规范编号，如 "GB 55037-2022"
    regulation_name: str  # 规范名称，如 "建筑防火通用规范"
    chapter_num: str  # 大章编号，如 "2"
    chapter_title: str  # 大章标题，如 "基本规定"
    section_num: str  # 小节编号，如 "2.2"
    section_title: str  # 小节标题，如 "消防救援设施"
    clause_num: str  # 条款编号，如 "2.2.3"
    text: str  # 条款正文（含子项）
    sub_items: list[str] = field(default_factory=list)  # 子项列表
    explanation: str = ""  # 条文说明

    # 用于打印时的格式化引用 ---返回---> 《建筑防火通用规范》(GB 55037-2022) 第2章 基本规定 第2.2节 消防救援设施 第2.2.3条
    def citation(self) -> str:
        """
        生成可读的引用串
        """
        parts = [f"《{self.regulation_name}》({self.regulation_code})"]
        if self.chapter_title:
            parts.append(f"第{self.chapter_num}章 {self.chapter_title}")
        if self.section_title and self.section_num != self.clause_num:
            parts.append(f"第{self.section_num}节 {self.section_title}")
        parts.append(f"第{self.clause_num}条")
        return " ".join(parts)


# ------------------------------------------------------------------
# 文本抽取 （以<w:t[^>]*>(.*?)</w:t>为文字碎片）并清晰数据
# ------------------------------------------------------------------
# DOCX 的本质是一个 ZIP 压缩包，后缀改为 .zip 就能解压。里面 word/document.xml 是真实的文字内容
# ------------------------------------------------------------------
def _extract_docx_paragraphs(path: str) -> list[str]:
    """从 docx 中抽取段落文本（保留段落边界）"""
    with zipfile.ZipFile(path) as z:
        with z.open("word/document.xml") as f:
            xml = f.read().decode("utf-8", errors="ignore")

    # 按段落切分（Word 为了处理样式，可能把一个段落的文字切成很多 <w:t> 碎片，先找到所有段落，再把段落里的所有文字碎片拼起来）
    # <w:p>...</w:p>：Word 里的一个段落
    para_iter = re.finditer(r"<w:p\b[^>]*>(.*?)</w:p>", xml, re.DOTALL)  # 找出所有段落，返回迭代器，节省内存
    # <w:t>...</w:t>：段落里的一小段文字
    t_re = re.compile(r"<w:t[^>]*>(.*?)</w:t>", re.DOTALL)  # 编译正则，方便后续使用，更简洁，节省编译计算资源

    paragraphs: list[str] = []
    for i in para_iter:  # for 循环去取迭代器内容，i为一段内容
        body = i.group(1)  # 取出一段内的捕获组内容
        text = "".join(t_re.findall(body))  # 取出一个段落捕获组里面的全部细分text捕获组
        # 合并字间多余空白（正文章节标题常出现 "基 本 规 定" 这种字间空格）
        text = text.strip()
        # 过滤纯页码行，如 "· 61 ·" / "61"
        if text and not re.fullmatch(r"[·\s\d]+", text):
            paragraphs.append(text)
    return paragraphs


# ------------------------------------------------------------------
# pdf 抽取 （以行为碎片）并清晰数据
# ------------------------------------------------------------------
def _extract_pdf_paragraphs(path: str) -> list[str]:
    """从 pdf 中抽取段落文本"""
    from pypdf import PdfReader

    reader = PdfReader(path)
    paragraphs: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        # PDF 抽取的文本按行拆分，过滤页码与空行
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # 过滤纯页码行，如 "· 61 ·" / "61"
            if re.fullmatch(r"[·\s\d]+", line):
                continue
            paragraphs.append(line)
    return paragraphs


# ------------------------------------------------------------------
# 文本工具
# ------------------------------------------------------------------

# 处理标题工具
def _normalize_title(sting_to_sub: str) -> str:
    """去除字间空白，用于标题长度判断与归一化"""
    return re.sub(r"\s+", "", sting_to_sub)


_PUNCT_END = re.compile(r"[。；;：:！!?？]$")

# 判断标题
def _is_chapter_heading(num: str, title: str) -> bool:
    """判断 "数字 +  文本" 是大章标题还是子项

    判据：
    1、编号在 1~15 之间，且为纯数字
    2、标题较短，去掉空格 ≤14 字
    3、不以标点结尾。

    子项一般是完整句子，长度普遍 > 14 字或以 ；。 结尾。
    """
    if not (num.isdigit() and 1 <= int(num) <= 15):
        return False
    norm = _normalize_title(title)
    if _PUNCT_END.search(norm):
        return False
    return len(norm) <= 14


# 正则
_RE_CHAPTER_OR_ITEM = re.compile(r"^\s*(\d+)\s+(\S.*)$")  # x?????
_RE_SECTION = re.compile(r"^\s*(\d+\.\d+)\s+(\S.*)$")  # x.x?????
_RE_CLAUSE = re.compile(r"^\s*(\d+\.\d+\.\d+)\s+(\S.*)$")  # x.x.x??????

# 正文/说明分界
_RE_EXPL = re.compile(r"条\s*文\s*说\s*明")

# 清楚文本中的中文字符之间的空格
def _clean_spaces(text: str) -> str:
    """去除中文字符之间的空格，保留换行和英文/数字间的必要空格"""
    # 1. 去除中文字符前后的所有空格（包括制表符等）
    text = re.sub(r'(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])', '', text)  # 中文之间
    text = re.sub(r'(?<=[\u4e00-\u9fff])[ \t]+(?=[，。；：、！？])', '', text)  # 中文与中文标点之间
    text = re.sub(r'(?<=[，。；：、！？])[ \t]+(?=[\u4e00-\u9fff])', '', text)  # 中文标点与中文之间
    # 2. 清理连续空格（保留英文单词间的单个空格）
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text

# 将正文章节段落解析为条款字典列表（不含条文说明）一个条款一个 dict
def _parse_body(paragraphs: list[str]) -> list[dict]:
    """
    将正文章节段落解析为条款字典列表（不含条文说明）
    返回一个个[{},{},{}……]: [{chapter_num, chapter_title, section_num, section_title,
            clause_num, text, sub_items}, ...]
    """
    cur_chapter_num = ""
    cur_chapter_title = ""
    cur_section_num = ""
    cur_section_title = ""
    cur_clause_num = ""
    cur_text = ""
    cur_sub_items: list[str] = []
    last_item_open = False  # 上一子项是否未以标点结束（待续行）
    out: list[dict] = []

    # 把当前条款存档
    def flush_clause():
        nonlocal cur_text, cur_sub_items, cur_clause_num, last_item_open
        if cur_clause_num and cur_text.strip():
            cleaned_text = _clean_spaces(cur_text.strip())
            out.append(
                {
                    "chapter_num": cur_chapter_num,
                    "chapter_title": cur_chapter_title,
                    "section_num": cur_section_num,
                    "section_title": cur_section_title,
                    "clause_num": cur_clause_num,
                    "text": cleaned_text,
                    "sub_items": list(cur_sub_items),
                }
            )
        cur_text = ""
        cur_sub_items = []
        last_item_open = False

    for line in paragraphs:
        m_i = _RE_CHAPTER_OR_ITEM.match(line)  # 章    x??????
        m_s = _RE_SECTION.match(line)  # 节    x.x??????
        m_c = _RE_CLAUSE.match(line)  # 条    x.x.x??????

        if m_c:  # 如果是条款 clause
            flush_clause()  # 存档
            cur_clause_num = m_c.group(1)  # 取条款号  取捕获组（1）
            cur_text = m_c.group(2).strip()  # 取条款具体内容  取捕获组（2）
            ch, sec, _ = cur_clause_num.split(".")  # 按 "." 拆分条款号
            if ch != cur_chapter_num:
                cur_chapter_num = ch  # 更新章
                cur_chapter_title = ""
            if sec != cur_section_num:  # 更新节
                cur_section_num = f"{ch}.{sec}"
                cur_section_title = ""
        elif m_s and len(_normalize_title(m_s.group(2))) <= 14:  # 如果是小节  section
            # 小节标题
            flush_clause()  # 存档
            cur_section_num = m_s.group(1)  # 取捕获组（1）
            cur_section_title = _normalize_title(m_s.group(2))  # 取捕获组（2）
            ch = cur_section_num.split(".")[0]
            if ch != cur_chapter_num:
                cur_chapter_num = ch
                cur_chapter_title = ""
        elif m_i and _is_chapter_heading(m_i.group(1), m_i.group(2)):  # 如果是章标题  chapter
            # 大章标题
            flush_clause()  # 存档
            cur_chapter_num = m_i.group(1)  # 取捕获组（1）
            cur_chapter_title = _normalize_title(m_i.group(2))  # 取捕获组（2）
            cur_section_num = ""
            cur_section_title = ""
        elif cur_clause_num and m_i:  # 在某条款下，且单个数字不为空，则为子项
            # 子项（非章节标题的 "<数字> <文本>"）
            item_text = m_i.group(2).strip()
            cur_sub_items.append(item_text)
            cur_text += f"\n{m_i.group(1)} {item_text}"
            last_item_open = not _PUNCT_END.search(item_text)
        elif cur_clause_num:
            # 普通续行：拼到当前子项或条款正文
            if cur_sub_items and last_item_open:
                cur_sub_items[-1] += line
                cur_text += line
                last_item_open = not _PUNCT_END.search(line)
            else:
                cur_text += line
        # else: 正文前的封面/目录等，跳过

    flush_clause()  # 存档
    return out


# 解析"条文说明"部分，返回 {clause_num: explanation_text}
def _parse_explanations(paragraphs: list[str]) -> dict[str, str]:
    """
    解析"条文说明"部分，返回 {clause_num: explanation_text}
    条文说明里 "x.x.x <text>" 即为对该条款的说明
    """
    explanations: dict[str, str] = {}
    cur_num = ""
    cur_text: list[str] = []

    def flush():
        nonlocal cur_num, cur_text
        if cur_num and cur_text:
            explanations[cur_num] = "\n".join(cur_text).strip()
        cur_num = ""
        cur_text = []

    for line in paragraphs:
        m_c = _RE_CLAUSE.match(line)
        if m_c:
            flush()
            cur_num = m_c.group(1)
            cur_text = [m_c.group(2).strip()]
            continue
        if cur_num:
            # 续行拼到当前说明
            cur_text.append(line)
    flush()
    return explanations


# ------------------------------------------------------------------
# 规范解析器：外部入口
# ------------------------------------------------------------------
def parse_regulation(path: str, regulation_code: str, regulation_name: str) -> list[Clause]:
    """
    解析规范文件，返回结构化条款列表

    Args:
        path: PDF / DOCX 文件路径
        regulation_code: 规范编号，如 "GB 55037-2022"
        regulation_name: 规范名称，如 "建筑防火通用规范"
    """

    # 判断后缀，调用不同的解析器
    tail = os.path.splitext(path)[1].lower()  # 取出后缀
    if tail == ".docx":
        paragraphs = _extract_docx_paragraphs(path)
    elif tail == ".pdf":
        paragraphs = _extract_pdf_paragraphs(path)
    else:
        raise ValueError(f"不支持的文件类型: {tail}（仅支持 .pdf / .docx）")

    # 定位正文 / 条文说明边界
    body_end = len(paragraphs)  # 先假设整个文件都是正文
    expl_start: int | None = None  # 假设未找到条文说明
    for i, p in enumerate(paragraphs):
        if expl_start is None and _RE_EXPL.search(p):
            body_end = i  # 正文结束点
            expl_start = i # 说明起始点
            break

    body_paras = paragraphs[:body_end]  # 找到正文内容结尾
    expl_paras = paragraphs[expl_start:] if expl_start is not None else []  # 找到解释条文开始

    # 跳过前言（公告/废止条款列表/目录），从首个"1 总则"或"1.0.1"开始
    body_start = 0
    for i, p in enumerate(body_paras):
        if re.match(r"^1\s+总\s*则", p) or re.match(r"^1\.0\.1\s", p):
            body_start = i
            break

    body_paras = body_paras[body_start:]  # 切片切出正文内容

    body = _parse_body(body_paras)  # 把条款打包成字典列表 list[dict, dict......]
    explanations = _parse_explanations(expl_paras)  # 把说明条文打包成dict[str, str......]

    clauses: list[Clause] = []
    for clause in body:  # 加入规范编号和规范名称
        clauses.append(
            Clause(
                regulation_code=regulation_code,
                regulation_name=regulation_name,
                chapter_num=clause["chapter_num"],
                chapter_title=clause["chapter_title"],
                section_num=clause["section_num"],
                section_title=clause["section_title"],
                clause_num=clause["clause_num"],
                text=clause["text"],
                sub_items=clause["sub_items"],
                explanation=explanations.get(clause["clause_num"], ""),
            )
        )
    return clauses  # 输出最终处理好的条款 list[dict, dict, dict......]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python pdf_parser.py <regulation.pdf|docx>")
        sys.exit(1)

    cs = parse_regulation(
        sys.argv[1],
        regulation_code="GB 55037-2022",
        regulation_name="建筑防火通用规范",
    )
    print(f"共解析条款 {len(cs)} 条")
    for c in cs[:5]:
        print("----")
        print(c.citation())
        print(c.text[:120])
