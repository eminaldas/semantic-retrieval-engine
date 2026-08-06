import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from neo4j import GraphDatabase
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.query import step5_query_engine as qe
from backend.query import step5c_risk_ranking as risk_ranking
from backend.query import step5b_entity_linker
from backend.pipeline import step1_chunker
from backend.management import manage_sources
from backend.query import step5d_conversation_memory as conv
from backend.shared import gemini_usage
from backend.shared.paths import PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(BASE_DIR, "artifacts", "chunks.jsonl")
EXTRACTION_FILE = os.path.join(BASE_DIR, "artifacts", "extraction.jsonl")
SELF_CORRECTION_LOG_FILE = os.path.join(BASE_DIR, "artifacts", "self_correction_log.jsonl")
ANSWERER_PROMPT_FILE = os.path.join(BASE_DIR, "docs", "prompts", "02_answerer_prompt.md")
LINKER_PROMPT_FILE = os.path.join(BASE_DIR, "docs", "prompts", "07_entity_linker_prompt.md")
MODE_DETECTION_PROMPT_FILE = os.path.join(BASE_DIR, "docs", "prompts", "11_mode_detection_prompt.md")
HISTORY_FILE = os.path.join(BASE_DIR, "artifacts", "conversation_history.jsonl")
SUMMARY_FILE = os.path.join(BASE_DIR, "artifacts", "conversation_summary.json")
REWRITE_PROMPT_FILE = os.path.join(BASE_DIR, "docs", "prompts", "09_conversation_rewrite_prompt.md")
SUMMARY_PROMPT_FILE = os.path.join(BASE_DIR, "docs", "prompts", "10_conversation_summary_prompt.md")
DASHBOARD_HTML_FILE = os.path.join(MODULE_DIR, "dashboard.html")
FRONTEND_DIST_DIR = os.path.join(BASE_DIR, "frontend", "dist")
FRONTEND_INDEX_FILE = os.path.join(FRONTEND_DIST_DIR, "index.html")
SOURCES_DIR = os.path.join(BASE_DIR, "sources")
UPLOAD_STEPS = [
    ("backend.pipeline.step1_chunker", "Chunking"),
    ("backend.pipeline.step2_extractor", "Çıkarım yapılıyor"),
    ("backend.pipeline.step3_graph_writer", "Graf yazılıyor"),
    ("backend.pipeline.step4_vector_writer", "Vektör yazılıyor"),
    ("backend.pipeline.step6_self_correction", "Kendini-düzeltme"),
]


def build_graph_response(node_rows, edge_rows):
    """docs/superpowers/specs/2026-07-23-dashboard-design.md #3 (/api/graph)."""
    nodes = [{"id": r["id"], "label": r["name"], "group": r["type"]} for r in node_rows]
    edges = [
        {"from": r["from_id"], "to": r["to_id"], "label": r["rel_type"], "status": r["status"]}
        for r in edge_rows
    ]
    return {"nodes": nodes, "edges": edges}


def read_self_correction_log(path):
    """Dosya yoksa bos liste (henuz self-correction calistirilmamis olabilir)."""
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_pipeline_status(chunks, entity_count, relation_count, qdrant_chunks_count, qdrant_entities_count):
    source_files = sorted({c["source_id"] for c in chunks})
    return {
        "chunk_count": len(chunks),
        "entity_count": entity_count,
        "relation_count": relation_count,
        "qdrant_chunks_points": qdrant_chunks_count,
        "qdrant_entities_points": qdrant_entities_count,
        "source_files": source_files,
    }


def build_sources_response(chunk_groups, extraction_groups):
    """manage_sources.py::group_chunks_by_source/group_extraction_by_source ciktisini
    GET /api/sources icin JSON-serilestirilebilir bir listeye cevirir. cmd_list'in
    print mantiginin JSON karsiligi - yeni mantik yok."""
    result = []
    for source_id in sorted(chunk_groups):
        c = chunk_groups[source_id]
        e = extraction_groups.get(source_id, {"entity_count": 0, "relation_count": 0})
        result.append({
            "source_id": source_id,
            "chunk_count": c["chunk_count"],
            "entity_count": e["entity_count"],
            "relation_count": e["relation_count"],
            "ingested_at": c["ingested_at"],
        })
    return result


def build_source_content_response(chunks):
    """source_id'ye ait chunk kayitlarini chunk_id'ye gore siralayip
    GET /api/sources/{source_id}/content icin JSON listesine cevirir."""
    sorted_chunks = sorted(chunks, key=lambda c: c["chunk_id"])
    return [{"chunk_id": c["chunk_id"], "location": c["location"], "text": c["text"]} for c in sorted_chunks]


def build_pair_structural(candidate_paths, best_path):
    """MODE_B (pair) icin structural alani - path listesi + en iyi path."""
    return {
        "candidate_paths": [
            {"score": p["score"], "formatted": qe.format_path(p)} for p in candidate_paths
        ],
        "best_path_formatted": qe.format_path(best_path) if best_path else None,
    }


def build_open_structural(ranked_nodes):
    """MODE_A (open) icin structural alani - risk siralamasi (entity_id + skor)."""
    return {
        "ranked_nodes": [{"entity_id": n["entity_id"], "score": n["score"]} for n in ranked_nodes]
    }


def create_session_token(secret):
    """Suresi olan, HMAC ile imzalanmis bir oturum tokeni uretir. Veritabani/dosya gerekmez -
    dogrulama sadece secret + tokenin kendisiyle yapilir."""
    issued_at = str(int(time.time()))
    signature = hmac.new(secret.encode(), issued_at.encode(), hashlib.sha256).hexdigest()
    return f"{issued_at}.{signature}"


def verify_session_token(token, secret, max_age_seconds=7 * 24 * 3600):
    """create_session_token'in urettigi tokeni dogrular - imza yanlissa, format bozuksa ya da
    sure dolmussa (ya da gelecekte olusturulmus gibi gorunuyorsa) False doner."""
    try:
        issued_at, signature = token.split(".", 1)
    except ValueError:
        return False
    expected = hmac.new(secret.encode(), issued_at.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return False
    try:
        age = time.time() - int(issued_at)
    except ValueError:
        return False
    return 0 <= age < max_age_seconds


def build_entity_link_error_detail(failures):
    """step5b_entity_linker.EntityLinkError.failures -> HTTPException detail sozlugu.
    Kullaniciya CLI'nin ham aday listesi yerine anlasilir, kisa bir mesaj gosterilir."""
    return {
        "message": "Sorunuzdaki varlıkları tanıyamadım, lütfen daha açık ifade eder misiniz?",
        "failures": [{"role": f["role"], "phrase": f["phrase"]} for f in failures],
    }


def build_query_response(mode, structural, ranked_evidence, answer, citation_issues, answer_error=None):
    """mode: 'pair'/'open'/'general'. structural: build_pair_structural/build_open_structural'in
    ciktisi (pair/open icin) ya da None (general icin - o modda yapisal bir kavram yok)."""
    return {
        "mode": mode,
        "structural": structural,
        "evidence": [
            {
                "sid": f"S{i}",
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "location": chunk["location"],
                "quote": chunk["quote"],
                "text": chunk["text"],
            }
            for i, (chunk, score) in enumerate(ranked_evidence, start=1)
        ],
        "answer": (
            {
                "answer_text": answer.answer_text,
                "risk_level": answer.risk_level,
                "used_sources": answer.used_sources,
                "assumptions": answer.assumptions,
            }
            if answer is not None else None
        ),
        "citation_issues": citation_issues,
        "answer_error": answer_error,
    }


def build_conversation_hydration_response(history):
    """conversation_history.jsonl kayitlarini frontend'in {id, question, response}
    seklindeki turns state'ine donusturur. CLI kokenli turlarda (response alani None)
    minimal ama gecerli bir response sentezlenir - AnswerCard.jsx'in response.answer/
    response.evidence/response.answer_error destructure'iyla uyumlu, uydurma veri degil,
    "bu soru icin detayli kanit/risk verisi yok" gerceginin durust temsili."""
    hydrated = []
    for turn in history:
        response = turn.get("response")
        if response is None:
            response = {
                "mode": turn["mode"],
                "structural": None,
                "evidence": [],
                "answer": {
                    "answer_text": turn["answer_text"],
                    "risk_level": None,
                    "used_sources": [],
                    "assumptions": [],
                },
                "citation_issues": [],
                "answer_error": None,
            }
        hydrated.append({"id": turn["turn_id"], "question": turn["question"], "response": response})
    return hydrated


def run_upload_job(job_id, file_path, source_id, chunks_file, extraction_file, neo4j_driver, qdrant_client):
    """Arka planda calisir (FastAPI BackgroundTasks). app.state.upload_jobs[job_id]'i
    her adimdan once/sonra gunceller - POST /api/sources/upload/status/{job_id} bunu okur.
    Canli-servis kodu, unit test edilmiyor (bkz. plan Global Constraints)."""
    total = len(UPLOAD_STEPS)
    existing_chunks = manage_sources.load_jsonl(chunks_file)
    already_exists = any(c["source_id"] == source_id for c in existing_chunks)

    if already_exists:
        app.state.upload_jobs[job_id] = {
            "status": "running", "step": 0, "total_steps": total,
            "step_name": "Güncelleniyor (eski veri siliniyor)", "source_id": source_id,
            "error_message": None, "result": None,
        }
        manage_sources.remove_source(source_id, chunks_file, extraction_file, neo4j_driver, qdrant_client)

    for i, (step_module, step_label) in enumerate(UPLOAD_STEPS, start=1):
        app.state.upload_jobs[job_id] = {
            "status": "running", "step": i, "total_steps": total,
            "step_name": step_label, "source_id": source_id,
            "error_message": None, "result": None,
        }
        args = [sys.executable, "-m", step_module]
        if step_module == "backend.pipeline.step1_chunker":
            args.append(file_path)
        result = subprocess.run(args, cwd=BASE_DIR)
        if result.returncode != 0:
            app.state.upload_jobs[job_id] = {
                "status": "error", "step": i, "total_steps": total,
                "step_name": step_label, "source_id": source_id,
                "error_message": f"{step_module} basarisiz oldu (exit code {result.returncode})",
                "result": None,
            }
            return

    app.state.chunks_by_id = qe.load_chunks_by_id(chunks_file)
    chunks = list(app.state.chunks_by_id.values())
    extraction = manage_sources.load_jsonl(extraction_file)
    chunk_groups = manage_sources.group_chunks_by_source(chunks)
    extraction_groups = manage_sources.group_extraction_by_source(chunks, extraction)
    sources = build_sources_response(chunk_groups, extraction_groups)
    this_source = next((s for s in sources if s["source_id"] == source_id), None)

    app.state.upload_jobs[job_id] = {
        "status": "done", "step": total, "total_steps": total,
        "step_name": "Tamamlandı", "source_id": source_id,
        "error_message": None, "result": this_source,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    print("Modeller yukleniyor, Neo4j/Qdrant'a baglaniliyor...")
    app.state.chunks_by_id = qe.load_chunks_by_id(CHUNKS_FILE)
    app.state.embedding_model = SentenceTransformer(qe.EMBEDDING_MODEL_NAME)
    app.state.cross_encoder = CrossEncoder(qe.CROSS_ENCODER_MODEL_NAME)
    app.state.neo4j_driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD")),
    )
    app.state.qdrant_client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    app.state.genai_client = genai.Client(api_key=api_key) if api_key else None
    app.state.answerer_model = os.environ.get("GEMINI_ANSWERER_MODEL", qe.DEFAULT_ANSWERER_MODEL)
    app.state.answerer_prompt_template = qe.load_prompt_template(ANSWERER_PROMPT_FILE)
    app.state.linker_model = os.environ.get("GEMINI_EXTRACTOR_MODEL", step5b_entity_linker.DEFAULT_MODEL)
    app.state.linker_prompt_template = step5b_entity_linker.load_prompt_template(LINKER_PROMPT_FILE)
    app.state.mode_detection_prompt_template = qe.load_prompt_template(MODE_DETECTION_PROMPT_FILE)
    app.state.rewrite_prompt_template = conv.load_prompt_template(REWRITE_PROMPT_FILE)
    app.state.summary_prompt_template = conv.load_prompt_template(SUMMARY_PROMPT_FILE)
    app.state.upload_jobs = {}
    app.state.dashboard_password = os.environ.get("DASHBOARD_PASSWORD")
    app.state.session_secret = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)
    if not app.state.dashboard_password:
        print("UYARI: DASHBOARD_PASSWORD tanimli degil, dashboard KORUMASIZ calisiyor.")

    print("Dashboard hazir: http://localhost:8000")
    yield

    app.state.neo4j_driver.close()


class LoginRequest(BaseModel):
    password: str


app = FastAPI(lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST_DIR, "assets")), name="assets")


@app.post("/api/login")
def login(req: LoginRequest, response: Response):
    if not app.state.dashboard_password:
        raise HTTPException(status_code=400, detail="DASHBOARD_PASSWORD tanimli degil, giris gerekmiyor.")
    if not secrets.compare_digest(req.password, app.state.dashboard_password):
        raise HTTPException(status_code=401, detail="Sifre yanlis.")
    token = create_session_token(app.state.session_secret)
    response.set_cookie("session", token, httponly=True, samesite="lax", max_age=7 * 24 * 3600)
    return {"ok": True}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie("session")
    return {"ok": True}


@app.get("/api/session")
def get_session(request: Request):
    if not request.app.state.dashboard_password:
        return {"authenticated": True}
    token = request.cookies.get("session")
    authenticated = bool(token and verify_session_token(token, request.app.state.session_secret))
    return {"authenticated": authenticated}


def require_session(request: Request):
    if not request.app.state.dashboard_password:
        return
    token = request.cookies.get("session")
    if not token or not verify_session_token(token, request.app.state.session_secret):
        raise HTTPException(status_code=401, detail="Oturum gerekli.")


@app.get("/")
def index():
    return FileResponse(FRONTEND_INDEX_FILE)


@app.get("/admin")
def admin(request: Request):
    if request.app.state.dashboard_password:
        token = request.cookies.get("session")
        if not token or not verify_session_token(token, request.app.state.session_secret):
            return RedirectResponse("/")
    return FileResponse(DASHBOARD_HTML_FILE)


@app.get("/api/graph", dependencies=[Depends(require_session)])
def get_graph():
    try:
        with app.state.neo4j_driver.session() as session:
            node_rows = [dict(r) for r in session.run(
                "MATCH (e:Entity) RETURN e.entity_id AS id, e.name AS name, e.type AS type"
            )]
            edge_rows = [dict(r) for r in session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) RETURN a.entity_id AS from_id, b.entity_id AS to_id, "
                "type(r) AS rel_type, coalesce(r.status, 'ACTIVE') AS status"
            )]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j'ye baglanilamadi: {exc}")
    return build_graph_response(node_rows, edge_rows)


@app.get("/api/self-correction-log", dependencies=[Depends(require_session)])
def get_self_correction_log():
    return read_self_correction_log(SELF_CORRECTION_LOG_FILE)


@app.get("/api/pipeline-status", dependencies=[Depends(require_session)])
def get_pipeline_status():
    chunks = list(app.state.chunks_by_id.values())
    try:
        with app.state.neo4j_driver.session() as session:
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            relation_count = session.run("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS c").single()["c"]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j'ye baglanilamadi: {exc}")
    try:
        qdrant_chunks_count = app.state.qdrant_client.count(collection_name="chunks").count
        qdrant_entities_count = app.state.qdrant_client.count(collection_name="entities").count
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant'a baglanilamadi: {exc}")
    return build_pipeline_status(chunks, entity_count, relation_count, qdrant_chunks_count, qdrant_entities_count)


@app.get("/api/usage", dependencies=[Depends(require_session)])
def get_usage():
    return gemini_usage.aggregate_usage()


@app.get("/api/sources", dependencies=[Depends(require_session)])
def get_sources():
    chunks = manage_sources.load_jsonl(CHUNKS_FILE)
    extraction = manage_sources.load_jsonl(EXTRACTION_FILE)
    chunk_groups = manage_sources.group_chunks_by_source(chunks)
    extraction_groups = manage_sources.group_extraction_by_source(chunks, extraction)
    return build_sources_response(chunk_groups, extraction_groups)


@app.get("/api/conversation", dependencies=[Depends(require_session)])
def get_conversation():
    history = conv.load_history(HISTORY_FILE)
    return build_conversation_hydration_response(history)


@app.delete("/api/conversation", dependencies=[Depends(require_session)])
def delete_conversation():
    conv.clear_conversation(HISTORY_FILE, SUMMARY_FILE)
    return {"status": "cleared"}


@app.get("/api/sources/{source_id}/content", dependencies=[Depends(require_session)])
def get_source_content(source_id: str):
    matching_chunks = [c for c in app.state.chunks_by_id.values() if c["source_id"] == source_id]
    if not matching_chunks:
        raise HTTPException(status_code=404, detail=f"'{source_id}' bulunamadi.")
    return build_source_content_response(matching_chunks)


@app.post("/api/sources/upload", dependencies=[Depends(require_session)])
async def upload_source(file: UploadFile, background_tasks: BackgroundTasks):
    source_id = file.filename
    ext = os.path.splitext(source_id)[1].lower()
    if ext not in step1_chunker.SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Desteklenmeyen dosya turu: {ext}")
    if any(job["status"] == "running" for job in app.state.upload_jobs.values()):
        raise HTTPException(status_code=409, detail="Baska bir yukleme devam ediyor.")

    os.makedirs(SOURCES_DIR, exist_ok=True)
    file_path = os.path.join(SOURCES_DIR, source_id)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    job_id = str(uuid.uuid4())
    app.state.upload_jobs[job_id] = {
        "status": "running", "step": 0, "total_steps": len(UPLOAD_STEPS),
        "step_name": "Başlatılıyor", "source_id": source_id,
        "error_message": None, "result": None,
    }
    background_tasks.add_task(
        run_upload_job, job_id, file_path, source_id,
        CHUNKS_FILE, EXTRACTION_FILE, app.state.neo4j_driver, app.state.qdrant_client,
    )
    return {"job_id": job_id, "source_id": source_id}


@app.get("/api/sources/upload-status/{job_id}", dependencies=[Depends(require_session)])
def get_upload_status(job_id: str):
    job = app.state.upload_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id bulunamadi.")
    return job


@app.delete("/api/sources/{source_id}", dependencies=[Depends(require_session)])
def delete_source(source_id: str):
    result = manage_sources.remove_source(
        source_id, CHUNKS_FILE, EXTRACTION_FILE, app.state.neo4j_driver, app.state.qdrant_client,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"'{source_id}' bulunamadi.")
    app.state.chunks_by_id = qe.load_chunks_by_id(CHUNKS_FILE)
    return result


class QueryRequest(BaseModel):
    mode: str | None = None
    cause: str | None = None
    target: str | None = None
    question: str
    max_hops: int = 2
    top_risk: int = 5


@app.post("/api/query", dependencies=[Depends(require_session)])
def post_query(req: QueryRequest):
    effective_question = req.question
    if app.state.genai_client is not None:
        effective_question = conv.prepare_question(
            HISTORY_FILE, SUMMARY_FILE, req.question,
            app.state.genai_client, conv.DEFAULT_MODEL, app.state.rewrite_prompt_template,
        )

    mode_was_auto_detected = req.mode is None and not req.cause and not req.target
    detected_mode = None
    if mode_was_auto_detected:
        detected_mode = qe.detect_mode_from_question(
            effective_question, app.state.genai_client, app.state.linker_model, app.state.mode_detection_prompt_template,
        )

    mode = qe.resolve_mode(detected_mode if mode_was_auto_detected else req.mode, req.cause, req.target)
    if mode != "pair" and (req.cause or req.target):
        raise HTTPException(status_code=400, detail=f"mode={mode} ile cause/target birlikte verilemez (bu modda A/C onceden bilinmiyor).")

    try:
        seed_chunk_ids = qe.vector_seed_search(app.state.qdrant_client, app.state.embedding_model, effective_question, 20)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant'a baglanilamadi: {exc}")

    if mode == "pair":
        cause_id, target_id = req.cause, req.target
        if not cause_id or not target_id:
            if app.state.genai_client is None:
                raise HTTPException(status_code=422, detail={"message": "Bu mod icin GEMINI_API_KEY gerekli."})
            try:
                link_result = step5b_entity_linker.link_entities(
                    effective_question, app.state.qdrant_client, app.state.embedding_model,
                    app.state.genai_client, app.state.linker_model, app.state.linker_prompt_template,
                )
                cause_id, target_id = link_result.cause_entity_id, link_result.target_entity_id
            except step5b_entity_linker.EntityLinkError as exc:
                if not mode_was_auto_detected:
                    raise HTTPException(status_code=422, detail=build_entity_link_error_detail(exc.failures))
                mode = "general"

    if mode == "pair":
        try:
            with app.state.neo4j_driver.session() as session:
                candidate_paths = qe.find_candidate_paths(session, cause_id, target_id, req.max_hops)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Neo4j'ye baglanilamadi: {exc}")
        evidence_pool = qe.build_evidence_pool(candidate_paths, seed_chunk_ids, app.state.chunks_by_id)
        best_path = candidate_paths[0] if candidate_paths else None
        structural = build_pair_structural(candidate_paths, best_path)
    elif mode == "open":
        try:
            with app.state.neo4j_driver.session() as session:
                ranked_nodes = risk_ranking.compute_risk_ranking(session, qe.RELATION_WEIGHTS, top_k=req.top_risk)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Neo4j'ye baglanilamadi: {exc}")
        evidence_pool = risk_ranking.build_risk_evidence_pool(ranked_nodes, seed_chunk_ids, app.state.chunks_by_id)
        best_path = None
        structural = build_open_structural(ranked_nodes)
    else:  # general
        evidence_pool = [app.state.chunks_by_id[cid] for cid in seed_chunk_ids if cid in app.state.chunks_by_id]
        best_path = None
        structural = None

    ranked_evidence = qe.rerank_evidence(app.state.cross_encoder, effective_question, evidence_pool)

    answer = None
    citation_issues = []
    answer_error = None
    if app.state.genai_client is None:
        answer_error = "GEMINI_API_KEY tanimli degil, Answerer atlaniyor."
    elif not ranked_evidence:
        answer_error = "Kanit bulunamadi, Answerer atlaniyor."
    else:
        try:
            context_pack = qe.build_context_pack(effective_question, best_path, ranked_evidence)
            answer = qe.call_answerer(
                app.state.genai_client, app.state.answerer_model,
                app.state.answerer_prompt_template, context_pack,
            )
            available_sids = [f"S{i}" for i in range(1, len(ranked_evidence) + 1)]
            citation_issues = qe.validate_citations(answer, available_sids)
        except Exception as exc:
            answer_error = f"Gemini API hatasi: {exc}"

    response = build_query_response(mode, structural, ranked_evidence, answer, citation_issues, answer_error)

    if answer is not None:
        conv.record_turn(
            HISTORY_FILE, SUMMARY_FILE,
            {"mode": mode, "question": req.question, "rewritten_question": effective_question,
             "answer_text": answer.answer_text, "response": response, "source": "gui"},
            app.state.genai_client, conv.DEFAULT_MODEL, app.state.summary_prompt_template,
        )

    return response


if __name__ == "__main__":
    import uvicorn

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    if not os.environ.get("NEO4J_PASSWORD"):
        print("HATA: NEO4J_PASSWORD tanimli degil. .env dosyasina bir sifre yaz (bkz. .env.example)")
        sys.exit(1)

    uvicorn.run("backend.api.dashboard:app", host="127.0.0.1", port=8000, reload=False)
