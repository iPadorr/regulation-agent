import json
import re


def parse_json(input_str):
    """
    解析 JSON 字符串，支持纯 JSON 格式和 Markdown 代码块格式
    """

    try:
        # 首先尝试直接解析纯 JSON 字符串（去除首尾空白）
        return json.loads(input_str.strip())
    except Exception:
        # 如果直接解析失败，尝试从 Markdown 代码块中提取 JSON
        # 使用正则表达式匹配 ```json ... ``` 格式的代码块
        pattern = r"```json\s*([\s\S]*?)\s*```"  # 匹配````json`开头，中间所有内容，`?`尽量少匹配，````结束
        match = re.findall(pattern, input_str)[0]

        # 避免没有匹配到JSON的报错
        if not match:
            raise ValueError("未找到有效的JSON")

        # 解析提取出的 JSON 内容
        return json.loads(match)


if __name__ == "__main__":
    print(
        parse_json("""
    ```json
    {
        "foo": "bar",
        "bar": "foo"
    }
    ```
    """)
    )

    print(parse_json("[1,2,3]"))
