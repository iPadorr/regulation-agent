"""API 路由注册"""
from fastapi import APIRouter

from .regulation import regulation_router  # 加载时跳到regulation创建方法的路由

api_router = APIRouter(prefix="/api")
api_router.include_router(regulation_router, prefix="/regulation")
