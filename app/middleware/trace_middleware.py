import uuid

from fastapi import Request

from app.util import request_id, trace_id


async def gen_trace_req_id(request: Request, call_next):
    """
    自动生成 request_id 和 trace_id 的中间件
    """

    # 生成或获取 trace_id
    incoming_trace = request.headers.get("X-Trace-ID")
    if incoming_trace:
        trace_token = trace_id.set(incoming_trace)  # 说明别的服务调用过来的，直接复用上游的 trace_id
    else:
        trace_token = trace_id.set(str(uuid.uuid4()))  # 用户直接调用，生成新的 trace_id

    # 生成 request_id
    request_token = request_id.set(str(uuid.uuid4()))

    # 处理请求
    response = await call_next(request)

    # 在响应头中返回这些ID，方便前端调试
    response.headers["X-Trace-ID"] = trace_id.get()
    response.headers["X-Request-ID"] = request_id.get()

    # 清除上下文
    trace_id.reset(trace_token)
    request_id.reset(request_token)

    return response
