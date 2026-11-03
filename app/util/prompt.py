from importlib import resources

import yaml
from jinja2 import Template


def get_prompt(prompt_file: str, prompt_name: str, **kwargs):
    """
    从 YAML 文件中读取提示词模板，并进行变量验证和渲染
    """

    # 使用 importlib.resources 从包中读取提示词文件。避免打包了找不到文件
    # 找到文件根目录，拼接文件名和后缀，读取成字符串，翻译成字典，读取prompt_name
    prompt_data = yaml.safe_load(
        resources.files("app.prompt").joinpath(f"{prompt_file}.yml").read_text()
    )[prompt_name]

    required_vars = prompt_data["required_vars"]  # 获取参数列表

    missing_vars = [var for var in required_vars if var not in kwargs]  # 验证必需的模板变量是否都已提供
    if missing_vars:
        error_msg = f"missing prompt variables: {missing_vars}"
        raise ValueError(error_msg)

    # 使用 Jinja2 模板引擎渲染系统提示词和用户提示词，返回字符串类型
    system_prompt = Template(prompt_data["system_template"]).render(**kwargs)
    user_prompt = Template(prompt_data["user_template"]).render(**kwargs)

    return {"system": system_prompt, "user": user_prompt}


if __name__ == "__main__":
    from pprint import pprint

    pprint(
        get_prompt(
            "regulation_rag",
            "regulation_rag",
            query="2222222",
            references="3333333",
        )
    )
