import uuid
import asyncio
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Body
from pydantic import BaseModel
import uvicorn

from src.compliance_analyzer import ComplianceAnalyzer, ReportGenerator, Config


PROTOCOLS_DB_PATH = Config.DATA_DIR / "protocols_db.json"



class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    job_id: str
    job_type: str
    protocol_id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ProtocolInfo(BaseModel):
    protocol_id: str
    name: str
    file_path: str
    doc_id: int
    indexed_at: datetime
    file_size: int



jobs: Dict[str, JobInfo] = {}
protocols: Dict[str, ProtocolInfo] = {}
executor = ThreadPoolExecutor(max_workers=4)
analyzer: Optional[ComplianceAnalyzer] = None


def save_protocols_db():
    """Save protocols to persistent storage"""
    data = {
        pid: {
            "protocol_id": info.protocol_id,
            "name": info.name,
            "file_path": info.file_path,
            "doc_id": info.doc_id,
            "indexed_at": info.indexed_at.isoformat(),
            "file_size": info.file_size
        }
        for pid, info in protocols.items()
    }
    PROTOCOLS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROTOCOLS_DB_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_protocols_db():
    """Load protocols from persistent storage"""
    global protocols
    if not PROTOCOLS_DB_PATH.exists():
        return

    try:
        with open(PROTOCOLS_DB_PATH, "r") as f:
            data = json.load(f)

        for protocol_id, info in data.items():
            if not Path(info["file_path"]).exists():
                print(f"Warning: Missing file: {info['file_path']}")
                continue

            protocols[protocol_id] = ProtocolInfo(
                protocol_id=info["protocol_id"],
                name=info["name"],
                file_path=info["file_path"],
                doc_id=info["doc_id"],
                indexed_at=datetime.fromisoformat(info["indexed_at"]),
                file_size=info["file_size"]
            )
        print(f"Loaded {len(protocols)} protocols")
    except Exception as e:
        print(f"Error loading protocols: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown"""
    global analyzer

    Config.PROTOCOLS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    try:
        analyzer = ComplianceAnalyzer(use_agent=Config.validate())
        print(f"Analyzer initialized (agent={'enabled' if analyzer.use_agent else 'disabled'})")
    except Exception as e:
        print(f"Analyzer init error: {e}")
        analyzer = None

    # Load and re-index protocols
    load_protocols_db()
    if analyzer and protocols:
        print("Re-indexing protocols...")
        max_doc_id = -1
        for pid, info in protocols.items():
            try:
                content = analyzer.load_document(Path(info.file_path))
                analyzer.vector_store.add_document(
                    doc_id=info.doc_id,
                    content=content,
                    metadata={"protocol_id": pid, "name": info.name, "file_path": info.file_path}
                )
                max_doc_id = max(max_doc_id, info.doc_id)
            except Exception as e:
                print(f"  Error re-indexing {info.name}: {e}")
        analyzer.protocol_counter = max_doc_id + 1

    yield

    executor.shutdown(wait=False)


def get_analyzer() -> ComplianceAnalyzer:
    """Dependency: get analyzer instance"""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    return analyzer


app = FastAPI(
    title="Clinical Trial Compliance API",
    description="KPI compliance analysis for clinical trial protocols",
    version="2.0.0",
    lifespan=lifespan,
)


def run_kpi_analysis(
    job_id: str,
    protocol_id: str,
    protocol_path: Path,
    requirements_content: bytes,
    requirements_filename: str,
    kpis: List[str]
):
    """Background: Run KPI compliance analysis"""
    global jobs
    jobs[job_id].status = JobStatus.PROCESSING

    try:
        protocol_info = protocols.get(protocol_id)
        if not protocol_info:
            raise ValueError(f"Protocol {protocol_id} not found")

        if analyzer.agent:
            analyzer.agent._current_doc_id = protocol_info.doc_id

        with tempfile.TemporaryDirectory() as tmp_dir:
            requirements_path = Path(tmp_dir) / requirements_filename
            requirements_path.write_bytes(requirements_content)

            results = analyzer.analyze_kpi_compliance(
                protocol_path=protocol_path,
                requirements_path=requirements_path,
                kpis=kpis
            )

        jobs[job_id].status = JobStatus.COMPLETED
        jobs[job_id].completed_at = datetime.now()
        jobs[job_id].result = results

    except Exception as e:
        jobs[job_id].status = JobStatus.FAILED
        jobs[job_id].completed_at = datetime.now()
        jobs[job_id].error = str(e)



@app.post("/protocols")
async def upload_protocol(
    file: UploadFile = File(...),
    current_analyzer: ComplianceAnalyzer = Depends(get_analyzer)
):
    """Upload and index a protocol document"""
    filename = file.filename or "protocol"
    suffix = Path(filename).suffix.lower()

    if suffix not in [".pdf", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    protocol_id = str(uuid.uuid4())
    save_path = Config.PROTOCOLS_DIR / f"{protocol_id}{suffix}"

    try:
        content = await file.read()
        save_path.write_bytes(content)

        protocol_content = current_analyzer.load_document(save_path)
        doc_id = current_analyzer.protocol_counter
        current_analyzer.protocol_counter += 1

        current_analyzer.vector_store.add_document(
            doc_id=doc_id,
            content=protocol_content,
            metadata={"protocol_id": protocol_id, "name": filename, "file_path": str(save_path)}
        )

        protocols[protocol_id] = ProtocolInfo(
            protocol_id=protocol_id,
            name=filename,
            file_path=str(save_path),
            doc_id=doc_id,
            indexed_at=datetime.now(),
            file_size=len(content)
        )
        save_protocols_db()

        return {
            "protocol_id": protocol_id,
            "name": filename,
            "doc_id": doc_id,
            "indexed_at": protocols[protocol_id].indexed_at.isoformat(),
            "file_size": len(content),
            "message": "Protocol uploaded successfully"
        }

    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.get("/protocols")
async def list_protocols():
    """List all protocols"""
    return {
        "count": len(protocols),
        "protocols": [
            {
                "protocol_id": p.protocol_id,
                "name": p.name,
                "doc_id": p.doc_id,
                "indexed_at": p.indexed_at.isoformat(),
                "file_size": p.file_size
            }
            for p in protocols.values()
        ]
    }


@app.get("/protocols/{protocol_id}")
async def get_protocol(protocol_id: str):
    """Get protocol details"""
    if protocol_id not in protocols:
        raise HTTPException(status_code=404, detail="Protocol not found")

    p = protocols[protocol_id]
    return {
        "protocol_id": p.protocol_id,
        "name": p.name,
        "doc_id": p.doc_id,
        "file_path": p.file_path,
        "indexed_at": p.indexed_at.isoformat(),
        "file_size": p.file_size
    }


@app.delete("/protocols/{protocol_id}")
async def delete_protocol(
    protocol_id: str,
    current_analyzer: ComplianceAnalyzer = Depends(get_analyzer)
):
    """Delete a protocol"""
    if protocol_id not in protocols:
        raise HTTPException(status_code=404, detail="Protocol not found")

    info = protocols[protocol_id]

    try:
        current_analyzer.vector_store.delete_document(info.doc_id)
        Path(info.file_path).unlink(missing_ok=True)
        del protocols[protocol_id]
        save_protocols_db()
        return {"message": "Protocol deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")


@app.post("/analyze/kpis")
async def analyze_kpis(
    requirements: UploadFile = File(...),
    protocol_id: str = Form(...),
    kpis: str = Form(...),
    current_analyzer: ComplianceAnalyzer = Depends(get_analyzer)
):
    """
    Submit KPI compliance analysis job.

    Workflow:
    - Step 3.1: Extract conditions from Protocol for each KPI
    - Step 3.2: Check conditions against Requirements
    - Step 3.3: Judge compliance

    Args:
        requirements: Requirements document (PDF/TXT/MD)
        protocol_id: ID of uploaded protocol
        kpis: Comma-separated list of KPIs
    """
    filename = requirements.filename or "requirements.txt"
    suffix = Path(filename).suffix.lower()

    if suffix not in [".pdf", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    if not current_analyzer.use_agent:
        raise HTTPException(status_code=400, detail="Agent mode required. Check API key.")

    if protocol_id not in protocols:
        raise HTTPException(status_code=404, detail="Protocol not found")

    kpis_list = [k.strip() for k in kpis.split(",") if k.strip()]
    if not kpis_list:
        raise HTTPException(status_code=400, detail="KPIs required")

    protocol_info = protocols[protocol_id]
    protocol_path = Path(protocol_info.file_path)

    if not protocol_path.exists():
        raise HTTPException(status_code=404, detail="Protocol file not found")

    requirements_content = await requirements.read()

    job_id = str(uuid.uuid4())
    jobs[job_id] = JobInfo(
        job_id=job_id,
        job_type="kpi_compliance_analysis",
        protocol_id=protocol_id,
        status=JobStatus.PENDING,
        created_at=datetime.now()
    )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        run_kpi_analysis,
        job_id, protocol_id, protocol_path,
        requirements_content, filename, kpis_list
    )

    return {
        "job_id": job_id,
        "protocol_id": protocol_id,
        "requirements_file": filename,
        "kpis_count": len(kpis_list),
        "kpis": kpis_list,
        "status": JobStatus.PENDING,
        "message": "Analysis started. Poll GET /jobs/{job_id} for status."
    }


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    response = {
        "job_id": job.job_id,
        "job_type": job.job_type,
        "protocol_id": job.protocol_id,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
    }

    if job.completed_at:
        response["completed_at"] = job.completed_at.isoformat()
    if job.status == JobStatus.COMPLETED:
        response["result"] = job.result
    elif job.status == JobStatus.FAILED:
        response["error"] = job.error

    return response


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get job result (completed jobs only)"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]:
        raise HTTPException(status_code=202, detail=f"Job {job.status}")
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Job failed: {job.error}")

    return {
        "job_id": job.job_id,
        "job_type": job.job_type,
        "protocol_id": job.protocol_id,
        "status": job.status,
        "result": job.result
    }


@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        "count": len(jobs),
        "jobs": [
            {
                "job_id": j.job_id,
                "job_type": j.job_type,
                "protocol_id": j.protocol_id,
                "status": j.status,
                "created_at": j.created_at.isoformat(),
                "completed_at": j.completed_at.isoformat() if j.completed_at else None
            }
            for j in jobs.values()
        ]
    }


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a completed/failed job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if jobs[job_id].status in [JobStatus.PENDING, JobStatus.PROCESSING]:
        raise HTTPException(status_code=400, detail="Cannot delete running job")

    del jobs[job_id]
    return {"message": "Job deleted"}



@app.post("/report/generate")
async def generate_report(analysis_result: Dict[str, Any] = Body(...)):
    """Generate markdown report from analysis result"""
    try:
        markdown = ReportGenerator.generate_markdown_report(analysis_result)
        return {"markdown": markdown}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")



@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "service": "Clinical Trial Compliance API",
        "version": "2.0.0",
        "agent_mode": analyzer.use_agent if analyzer else False
    }


@app.get("/system/info")
async def system_info():
    """System information"""
    return {
        "llm_provider": Config.LLM_PROVIDER,
        "embedding_provider": Config.EMBEDDING_PROVIDER,
        "protocols_dir": str(Config.PROTOCOLS_DIR),
        "agent_mode": analyzer.use_agent if analyzer else False,
        "protocols_count": len(protocols),
        "active_jobs": sum(1 for j in jobs.values() if j.status in [JobStatus.PENDING, JobStatus.PROCESSING])
    }


@app.delete("/system/clear")
async def clear_system(current_analyzer: ComplianceAnalyzer = Depends(get_analyzer)):
    """Clear all data"""
    global protocols, jobs

    try:
        current_analyzer.clear_vector_store()
        for info in protocols.values():
            Path(info.file_path).unlink(missing_ok=True)
        protocols = {}
        jobs = {}
        save_protocols_db()
        return {"message": "System cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {e}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
