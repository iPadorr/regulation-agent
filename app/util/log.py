import sys
from contextvars import ContextVar
from pathlib import Path

from loguru import logger

from app.config.conf import logging_config as conf_logging

# 定义上下文变量用于在异步环境中追踪 trace_id 和 request_id
# ContextVar 提供的协程级别隔离的变量，即使两个任务在同一线程上交替运行，它们也拥有该变量的独立副本
# 我们会在在请求处理的入口处（如 Web 中间件）设置这个变量的值


# 追踪从请求开始到结束的完整调用链路，贯穿多个服务、多个组件
trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
# 标识单个请求，只在当前服务或当前操作中有效
request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def setup_logging():
    """
    配置 loguru 日志器，支持动态上下文
    """
    # 避免重复配置
    if hasattr(logger, "_configured"):
        return

    # 移除默认处理器，避免重复输出
    # loguru 默认有一个输出到 stderr 的处理器，级别为 INFO
    logger.remove()

    # 动态上下文过滤器
    def context_filter(record):
        """
        动态获取当前上下文并注入到日志记录中
        """
        record["extra"]["trace_id"] = trace_id.get() or ""
        record["extra"]["request_id"] = request_id.get() or ""
        return record

    # 动态构建日志格式
    def get_log_format(record):
        """
        根据是否有上下文动态构建日志格式
        """
        # {time}: 时间戳，格式化为 YYYY-MM-DD HH:mm:ss
        # {level}: 日志级别，居中对齐8位宽度
        base_format = [
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green>",
            "<level>{level:^8}</level>",
        ]

        # {extra[trace_id]}: 追踪ID，用于跨服务追踪
        # {extra[request_id]}: 请求ID，用于标识单个请求
        trace_id = record["extra"].get("trace_id", "")
        request_id = record["extra"].get("request_id", "")

        if trace_id and request_id:
            base_format += [
                "<cyan>{extra[trace_id]}</cyan>:<cyan>{extra[request_id]}</cyan>"
            ]
        elif trace_id:
            base_format += [f"<cyan>{trace_id}</cyan>"]
        elif request_id:
            base_format += [f"<cyan>{request_id}</cyan>"]

        # {name}: 完整模块名
        # {function}: 函数名
        # {message}: 日志消息内容
        base_format += [
            "<cyan>{name}</cyan>.<cyan>{function}</cyan>",
            "<level>{message}</level>",
        ]

        return " | ".join(base_format) + "\n"

    # 控制台输出处理器
    if conf_logging.console_output:
        logger.add(
            sink=sys.stdout,  # 输出到标准输出
            format=get_log_format,  # 使用动态格式函数，根据上下文调整格式
            level=conf_logging.level,  # 从配置文件读取日志级别
            colorize=True,  # 启用颜色输出，提升可读性
            filter=context_filter,  # 应用动态上下文过滤器
            catch=False,  # 不捕获异常，让错误直接抛出
        )

    # 文件输出处理器
    if conf_logging.file_output:
        log_dir = Path(conf_logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)  # 确保日志目录存在

        logger.add(
            sink=log_dir / "{time:YYYY-MM-DD}.log",  # 按日期命名的日志文件
            format=get_log_format,  # 使用动态格式函数
            level=conf_logging.level,  # 使用配置的日志级别
            rotation=f"{conf_logging.max_file_size}",  # 按文件大小自动滚动
            encoding="utf-8",  # 使用UTF-8编码，支持中文
            filter=context_filter,  # 应用动态上下文过滤器
            catch=False,  # 不捕获异常，让错误直接抛出
        )

    # 标记为已配置
    setattr(logger, "_configured", True)


setup_logging()

if __name__ == "__main__":
    trace_id.set("trace-123")
    request_id.set("req-456")

    logger.info("info log")
    logger.warning("warning log")
    logger.error("error log")
    try:
        1 / 0
    except Exception:
        logger.exception("exception log")
