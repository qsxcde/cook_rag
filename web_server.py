"""
FastAPI Web 界面：加载 RecipeRAGSystem，提供聊天 API 与静态页。
启动: uvicorn web_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

# 保证可导入同目录下的 main、config
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from config import DEFAULT_CONFIG  # noqa: E402
from main import RecipeRAGSystem  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

rag: RecipeRAGSystem | None = None
init_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag, init_error
    init_error = None
    rag = None
    try:
        logger.info("正在加载 RAG 系统（嵌入模型与索引，可能较慢）…")
        rag = RecipeRAGSystem(DEFAULT_CONFIG)
        await asyncio.to_thread(rag.initialize_system)
        await asyncio.to_thread(rag.build_knowledge_base)
        logger.info("RAG 系统就绪")
    except Exception as e:
        init_error = str(e)
        rag = None
        logger.exception("RAG 初始化失败: %s", e)
    yield
    rag = None
    init_error = None


app = FastAPI(
    title="食谱 RAG",
    description="基于本地 Markdown 菜谱的 RAG 问答 Web 界面",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = _ROOT / "static"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    stream: bool = False
    session_id: str | None = None


def _ensure_rag():
    if init_error:
        raise HTTPException(status_code=503, detail=f"系统初始化失败: {init_error}")
    if rag is None or not rag.retrieval_module:
        raise HTTPException(status_code=503, detail="知识库未就绪，请稍后重试")


@app.get("/", response_class=HTMLResponse)
async def index_page():
    html_path = _static_dir / "index.html"
    if not html_path.is_file():
        return HTMLResponse("<h1>缺少 static/index.html</h1>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    if init_error:
        return {"ready": False, "error": init_error, "stats": None}
    if rag is None or not getattr(rag, "retrieval_module", None):
        return {"ready": False, "error": None, "stats": None}
    stats = None
    if rag.data_module and rag.data_module.documents:
        stats = rag.data_module.get_statistics()
    return {"ready": True, "error": None, "stats": stats}


@app.get("/api/sessions")
async def list_sessions():
    """获取所有历史会话列表"""
    _ensure_rag()
    assert rag is not None
    
    if not rag.conversation_manager:
        return {"sessions": []}
    
    sessions = rag.conversation_manager.list_sessions()
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """获取指定会话的详情（包含消息）"""
    _ensure_rag()
    assert rag is not None
    
    if not rag.conversation_manager:
        raise HTTPException(status_code=404, detail="会话管理未启用")
    
    session = rag.conversation_manager.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
    
    return {
        "session_id": session.session_id,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in session.messages
        ],
        "created_at": session.created_at,
        "updated_at": session.updated_at
    }


@app.post("/api/sessions")
async def create_session():
    """创建新会话"""
    _ensure_rag()
    assert rag is not None
    
    if not rag.conversation_manager:
        raise HTTPException(status_code=503, detail="会话管理未启用")
    
    session = rag.conversation_manager.create_session()
    return {"session_id": session.session_id}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    _ensure_rag()
    assert rag is not None
    
    if not rag.conversation_manager:
        raise HTTPException(status_code=503, detail="会话管理未启用")
    
    success = rag.conversation_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
    
    return {"success": True}


@app.post("/api/chat")
async def chat(body: ChatRequest):
    _ensure_rag()
    assert rag is not None

    if rag.conversation_manager:
        if body.session_id:
            rag.conversation_manager.load_session(body.session_id)
        elif not rag.conversation_manager.current_session:
            rag.conversation_manager.create_session()

    if body.stream:

        def token_stream():
            out = rag.ask_question(body.question.strip(), stream=True, use_conversation=True)
            if isinstance(out, str):
                yield out
                return
            for chunk in out:
                if chunk:
                    yield chunk

        return StreamingResponse(
            token_stream(),
            media_type="text/plain; charset=utf-8",
        )

    answer = await asyncio.to_thread(
        rag.ask_question, body.question.strip(), False, True
    )
    if not isinstance(answer, str):
        answer = "".join(answer)
    
    session_id = None
    if rag.conversation_manager and rag.conversation_manager.current_session:
        session_id = rag.conversation_manager.current_session.session_id
    
    return {"answer": answer, "session_id": session_id}


def main():
    import uvicorn

    uvicorn.run(
        "web_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
