import uuid
import asyncio
import json
import tempfile
import sqlite3
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


PROTOCOLS_DB_PATH = Config.DATA_DIR / "protocols.db"
HISTORY_DB_PATH = Config.DATA_DIR / "analysis_history.db"


def init_db():
    """Initialize SQLite database schema"""
    conn = sqlite3.connect(PROTOCOLS_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS protocols (
            protocol_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            doc_id INTEGER NOT NULL,
            indexed_at TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            content_hash TEXT UNIQUE
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_content_hash
        ON protocols(content_hash)
    """)

    conn.commit()
    conn.close()


def init_history_db():
    """Initialize history database schema"""
    conn = sqlite3.connect(HISTORY_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            protocol_name TEXT,
            requirements_name TEXT,
            topics_analyzed TEXT,
            total_topics INTEGER,
            matches INTEGER,
            partial_matches INTEGER,
            no_matches INTEGER,
            result_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()



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


class HistoryCreate(BaseModel):
    protocol_name: str
    requirements_name: str
    topics: List[str]
    results: Dict[str, Any]


class HistoryRecord(BaseModel):
    id: int
    timestamp: str
    protocol_name: Optional[str]
    requirements_name: Optional[str]
    topics_analyzed: List[str]
    total_topics: int
    matches: int
    partial_matches: int
    no_matches: int
    result_json: Optional[Dict[str, Any]] = None



jobs: Dict[str, JobInfo] = {}
protocols: Dict[str, ProtocolInfo] = {}
executor = ThreadPoolExecutor(max_workers=4)
analyzer: Optional[ComplianceAnalyzer] = None


def save_protocol_to_db(protocol: ProtocolInfo, content_hash: str):
    """Save single protocol to database"""
    conn = sqlite3.connect(PROTOCOLS_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO protocols 
        (protocol_id, name, file_path, doc_id, indexed_at, file_size, content_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        protocol.protocol_id,
        protocol.name,
        protocol.file_path,
        protocol.doc_id,
        protocol.indexed_at.isoformat(),
        protocol.file_size,
        content_hash
    ))
    
    conn.commit()
    conn.close()


def find_protocol_by_hash_db(content_hash: str) -> Optional[Dict[str, Any]]:
    """Find protocol by content hash in database"""
    conn = sqlite3.connect(PROTOCOLS_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT protocol_id, name, doc_id, indexed_at, file_size
        FROM protocols
        WHERE content_hash = ?
    """, (content_hash,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "protocol_id": row[0],
            "name": row[1],
            "doc_id": row[2],
            "indexed_at": row[3],
            "file_size": row[4]
        }
    return None


def load_protocols_db():
    """Load protocols from database"""
    global protocols
    
    if not PROTOCOLS_DB_PATH.exists():
        return

    try:
        conn = sqlite3.connect(PROTOCOLS_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT protocol_id, name, file_path, doc_id, indexed_at, file_size, content_hash
            FROM protocols
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            protocol_id, name, file_path, doc_id, indexed_at, file_size, content_hash = row
            
            if not Path(file_path).exists():
                print(f"Warning: Missing file: {file_path}")
                continue

            protocols[protocol_id] = ProtocolInfo(
                protocol_id=protocol_id,
                name=name,
                file_path=file_path,
                doc_id=doc_id,
                indexed_at=datetime.fromisoformat(indexed_at),
                file_size=file_size
            )
            
            # Rebuild hash map
            if content_hash and analyzer:
                analyzer.register_protocol_hash(content_hash, {
                    "protocol_id": protocol_id,
                    "name": name,
                    "doc_id": doc_id,
                    "indexed_at": indexed_at,
                    "file_size": file_size
                })
            
        print(f"Loaded {len(protocols)} protocols from database")
    except Exception as e:
        print(f"Error loading protocols: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown"""
    global analyzer

    Config.PROTOCOLS_DIR.mkdir(parents=True, exist_ok=True)
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize databases
    init_db()
    init_history_db()

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
    kpis: List[str],
    concurrent: bool = False
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
                kpis=kpis,
                concurrent=concurrent
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
    """Upload and index a protocol document (with deduplication)"""
    filename = file.filename or "protocol"
    suffix = Path(filename).suffix.lower()

    if suffix not in [".pdf", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    content = await file.read()
    content_hash = current_analyzer.compute_content_hash(content)

    # Check if protocol already exists in database
    existing = find_protocol_by_hash_db(content_hash)
    if existing:
        return {
            "protocol_id": existing["protocol_id"],
            "name": existing["name"],
            "doc_id": existing["doc_id"],
            "indexed_at": existing["indexed_at"],
            "file_size": existing["file_size"],
            "message": "Protocol already exists (reused existing)",
            "reused": True
        }

    # New protocol: save and index
    protocol_id = str(uuid.uuid4())
    save_path = Config.PROTOCOLS_DIR / f"{protocol_id}{suffix}"

    try:
        save_path.write_bytes(content)

        protocol_content = current_analyzer.load_document(save_path)
        doc_id = current_analyzer.protocol_counter
        current_analyzer.protocol_counter += 1

        current_analyzer.vector_store.add_document(
            doc_id=doc_id,
            content=protocol_content,
            metadata={
                "protocol_id": protocol_id,
                "name": filename,
                "file_path": str(save_path),
                "content_hash": content_hash
            }
        )

        protocols[protocol_id] = ProtocolInfo(
            protocol_id=protocol_id,
            name=filename,
            file_path=str(save_path),
            doc_id=doc_id,
            indexed_at=datetime.now(),
            file_size=len(content)
        )
        
        # Save to database and register hash
        save_protocol_to_db(protocols[protocol_id], content_hash)
        
        current_analyzer.register_protocol_hash(content_hash, {
            "protocol_id": protocol_id,
            "name": filename,
            "doc_id": doc_id,
            "indexed_at": protocols[protocol_id].indexed_at.isoformat(),
            "file_size": len(content)
        })

        return {
            "protocol_id": protocol_id,
            "name": filename,
            "doc_id": doc_id,
            "indexed_at": protocols[protocol_id].indexed_at.isoformat(),
            "file_size": len(content),
            "message": "Protocol uploaded successfully",
            "reused": False
        }

    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        import traceback
        print(f"Upload error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


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
        
        # Delete from database
        conn = sqlite3.connect(PROTOCOLS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM protocols WHERE protocol_id = ?", (protocol_id,))
        conn.commit()
        conn.close()
        
        del protocols[protocol_id]
        return {"message": "Protocol deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")


@app.post("/analyze/kpis")
async def analyze_kpis(
    requirements: UploadFile = File(...),
    protocol_id: str = Form(...),
    kpis: str = Form(...),
    concurrent: str = Form(default="false"),
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
        concurrent: Enable concurrent processing ("true"/"false")
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

    # Parse concurrent flag
    concurrent_mode = concurrent.lower() in ("true", "1", "yes")

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
        requirements_content, filename, kpis_list, concurrent_mode
    )

    return {
        "job_id": job_id,
        "protocol_id": protocol_id,
        "requirements_file": filename,
        "kpis_count": len(kpis_list),
        "kpis": kpis_list,
        "concurrent": concurrent_mode,
        "status": JobStatus.PENDING,
        "message": f"Analysis started {'(concurrent)' if concurrent_mode else '(sequential)'}. Poll GET /jobs/{job_id} for status."
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


# =============================================================================
# HISTORY ENDPOINTS
# =============================================================================

@app.post("/history")
async def create_history(data: HistoryCreate):
    """Save analysis results to history"""
    try:
        conn = sqlite3.connect(HISTORY_DB_PATH)
        cursor = conn.cursor()

        summary = data.results.get("summary", {})

        cursor.execute("""
            INSERT INTO analysis_history (
                timestamp, protocol_name, requirements_name, topics_analyzed,
                total_topics, matches, partial_matches, no_matches, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            data.protocol_name,
            data.requirements_name,
            json.dumps(data.topics),
            summary.get("total_requirements", 0),
            summary.get("followed", 0),
            summary.get("partial", 0),
            summary.get("not_followed", 0),
            json.dumps(data.results)
        ))

        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return {"id": record_id, "message": "History saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save history: {e}")


@app.get("/history")
async def get_history(limit: int = 50):
    """Get analysis history records"""
    try:
        conn = sqlite3.connect(HISTORY_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, timestamp, protocol_name, requirements_name, topics_analyzed,
                   total_topics, matches, partial_matches, no_matches
            FROM analysis_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        records = []
        for row in rows:
            topics = json.loads(row["topics_analyzed"]) if row["topics_analyzed"] else []
            records.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "protocol_name": row["protocol_name"],
                "requirements_name": row["requirements_name"],
                "topics_analyzed": topics,
                "total_topics": row["total_topics"],
                "matches": row["matches"],
                "partial_matches": row["partial_matches"],
                "no_matches": row["no_matches"]
            })

        return {"count": len(records), "records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {e}")


@app.get("/history/{record_id}")
async def get_history_record(record_id: int):
    """Get a single history record by ID"""
    try:
        conn = sqlite3.connect(HISTORY_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, timestamp, protocol_name, requirements_name, topics_analyzed,
                   total_topics, matches, partial_matches, no_matches, result_json
            FROM analysis_history
            WHERE id = ?
        """, (record_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Record not found")

        topics = json.loads(row["topics_analyzed"]) if row["topics_analyzed"] else []
        result_json = json.loads(row["result_json"]) if row["result_json"] else None

        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "protocol_name": row["protocol_name"],
            "requirements_name": row["requirements_name"],
            "topics_analyzed": topics,
            "total_topics": row["total_topics"],
            "matches": row["matches"],
            "partial_matches": row["partial_matches"],
            "no_matches": row["no_matches"],
            "result_json": result_json
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get record: {e}")


@app.delete("/history/{record_id}")
async def delete_history_record(record_id: int):
    """Delete a history record"""
    try:
        conn = sqlite3.connect(HISTORY_DB_PATH)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        if not deleted:
            raise HTTPException(status_code=404, detail="Record not found")

        return {"message": "Record deleted", "id": record_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete record: {e}")



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

        # Clear database
        conn = sqlite3.connect(PROTOCOLS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM protocols")
        conn.commit()
        conn.close()

        protocols = {}
        jobs = {}
        return {"message": "System cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {e}")


@app.post("/system/clear-cache")
async def clear_cache(current_analyzer: ComplianceAnalyzer = Depends(get_analyzer)):
    """Clear caches without deleting data (useful for memory management)"""
    try:
        current_analyzer.clear_caches()
        return {"message": "Caches cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {e}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
