import os
import uuid
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.data_engine import DataEngine
from backend.agents import MultiAgentOrchestrator
from backend.llm_service import LLMService
from backend.chart_builder import generate_chart_spec_from_tool_result, build_multi_tool_chart_spec, validate_chart_spec

app = FastAPI(title="InsightForge API", description="Conversational AI Data Scientist Engine")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(DATA_DIR, exist_ok=True)

# In-memory storage for persistent Investigation State & Orchestrator instances
INVESTIGATIONS_STORE: Dict[str, Dict[str, Any]] = {}
ORCHESTRATORS_STORE: Dict[str, MultiAgentOrchestrator] = {}

GLOBAL_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("LLM_API_KEY", ""))

class InvestigateRequest(BaseModel):
    dataset_name: str = "customer_churn.csv"
    research_question: str = "Why are customers leaving and how can we predict future churn?"
    api_key: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    investigation_id: str
    question: str
    history: Optional[List[ChatMessage]] = []
    api_key: Optional[str] = None

class ConfigKeyRequest(BaseModel):
    api_key: str

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "InsightForge Conversational Engine Operational", "llm_configured": bool(GLOBAL_API_KEY)}

@app.post("/api/config/key")
def set_api_key(req: ConfigKeyRequest):
    global GLOBAL_API_KEY
    GLOBAL_API_KEY = req.api_key.strip()
    return {"status": "ok", "configured": bool(GLOBAL_API_KEY)}

@app.get("/api/config/key")
def get_api_key_status():
    return {"configured": bool(GLOBAL_API_KEY)}

def sanitize_and_validate_path(filename: str, base_dir: str = DATA_DIR) -> str:
    """Sanitizes user-provided filename and verifies the path stays strictly within base_dir."""
    if not filename or not isinstance(filename, str):
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")
    
    clean_input = filename.strip()
    is_traversal_attempt = (".." in clean_input) or ("/" in clean_input) or ("\\" in clean_input) or (os.path.isabs(clean_input))
    
    safe_name = os.path.basename(clean_input.replace("\\", "/"))
    if not safe_name or safe_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    
    base_abs = os.path.abspath(base_dir)
    target_path = os.path.abspath(os.path.join(base_abs, safe_name))

    try:
        common = os.path.commonpath([base_abs, target_path])
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal attempt blocked: Invalid drive.")

    if common != base_abs or not target_path.startswith(base_abs):
        raise HTTPException(status_code=400, detail="Path traversal attempt blocked: Access denied.")

    if is_traversal_attempt and not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail="Path traversal attempt blocked: Invalid path manipulation.")

    return target_path

@app.get("/api/datasets")
def list_datasets():
    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    return {"datasets": files}

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    safe_filename = os.path.basename(file.filename.replace("\\", "/"))
    if not safe_filename or ".." in file.filename:
        raise HTTPException(status_code=400, detail="Path traversal attempt blocked in upload.")

    file_path = sanitize_and_validate_path(safe_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    data_engine = DataEngine(file_path)
    summary = data_engine.get_summary_profile()
    return {"message": "Dataset uploaded successfully", "summary": summary}

@app.post("/api/investigate")
def run_investigation(req: InvestigateRequest):
    global GLOBAL_API_KEY
    if req.api_key:
        GLOBAL_API_KEY = req.api_key.strip()

    try:
        file_path = sanitize_and_validate_path(req.dataset_name)
    except HTTPException:
        # Fallback to default sample dataset if specific requested name not found or invalid
        file_path = os.path.join(DATA_DIR, "customer_churn.csv")

    if not os.path.exists(file_path):
        file_path = os.path.join(DATA_DIR, "customer_churn.csv")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Requested dataset not found.")

    engine = DataEngine(file_path)
    orchestrator = MultiAgentOrchestrator(engine)
    results = orchestrator.run_investigation(req.research_question)
    
    investigation_id = f"inv_{str(uuid.uuid4())[:8]}"
    results["investigation_id"] = investigation_id
    results["chat_history"] = []
    
    INVESTIGATIONS_STORE[investigation_id] = results
    ORCHESTRATORS_STORE[investigation_id] = orchestrator
    
    return results

@app.post("/api/chat")
def chat_with_data(req: ChatRequest):
    global GLOBAL_API_KEY
    active_key = req.api_key or GLOBAL_API_KEY

    inv_id = req.investigation_id
    if inv_id not in INVESTIGATIONS_STORE:
        # If ID not found, attempt to grab latest active investigation
        if INVESTIGATIONS_STORE:
            inv_id = list(INVESTIGATIONS_STORE.keys())[-1]
        else:
            raise HTTPException(status_code=404, detail="No active investigation found. Run an investigation first.")

    state = INVESTIGATIONS_STORE[inv_id]
    orchestrator = ORCHESTRATORS_STORE.get(inv_id)
    llm_service = LLMService(api_key=active_key)

    formatted_history = [{"role": msg.role, "content": msg.content} for msg in (req.history or [])]

    # 1. Process conversational query with dynamic LLM Router
    route_info = llm_service.process_conversational_query(req.question, state, formatted_history)
    route = route_info.get("route", "GENERAL_CONVERSATION")

    answer_text = ""
    tool_executed = None
    chart_spec = None

    if route == "TARGETED_DATA_QUERY" and orchestrator:
        tool_name = route_info.get("tool_name")
        params = route_info.get("params", {})
        if tool_name:
            tool_res = orchestrator.execute_controlled_tool(tool_name, params)
            tool_executed = tool_name
            answer_text = llm_service.synthesize_tool_response(req.question, tool_name, tool_res, state, formatted_history)
            chart_spec = generate_chart_spec_from_tool_result(tool_name, tool_res)
        else:
            answer_text = route_info.get("answer", "No valid tool selected for data query.")

    elif route == "MULTI_TOOL_QUERY" and orchestrator:
        exec_res = orchestrator.execute_multi_tool_plan(route_info)
        if exec_res.get("success"):
            tool_executed = "multi_tool_plan"
            answer_text = llm_service.synthesize_multi_tool_response(req.question, exec_res, state, formatted_history)
            chart_spec = build_multi_tool_chart_spec(exec_res)
        else:
            tool_executed = "multi_tool_plan_failed"
            if exec_res.get("results"):
                answer_text = llm_service.synthesize_multi_tool_response(req.question, exec_res, state, formatted_history)
                chart_spec = build_multi_tool_chart_spec(exec_res)
            else:
                answer_text = f"Multi-tool execution failed: {exec_res.get('error', 'Invalid execution plan.')}"

    elif route == "FULL_INVESTIGATION_REQUEST" and orchestrator:
        research_q = route_info.get("research_question", req.question)
        new_results = orchestrator.run_investigation(research_q)
        new_results["investigation_id"] = inv_id
        INVESTIGATIONS_STORE[inv_id] = new_results
        state = new_results
        ai_ans = new_results.get("ai_answer", {})
        answer_text = ai_ans.get("direct_answer") or new_results.get("executive_summary", "Full investigation execution complete.")
        tool_executed = "full_investigation"

    else:  # GENERAL_CONVERSATION
        answer_text = route_info.get("answer") or "I don't have enough context from the current investigation to answer that reliably."

    # Update conversation history
    updated_history = formatted_history + [
        {"role": "user", "content": req.question},
        {"role": "assistant", "content": answer_text}
    ]
    state["chat_history"] = updated_history

    return {
        "investigation_id": inv_id,
        "answer": answer_text,
        "chat_history": updated_history,
        "tool_executed": tool_executed,
        "chart_spec": chart_spec
    }


@app.get("/api/investigation/{investigation_id}")
def get_investigation(investigation_id: str):
    if investigation_id not in INVESTIGATIONS_STORE:
        raise HTTPException(status_code=404, detail="Investigation not found.")
    return INVESTIGATIONS_STORE[investigation_id]

# Serve Frontend static assets
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_dashboard():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "InsightForge API active. Frontend index.html not found."}
