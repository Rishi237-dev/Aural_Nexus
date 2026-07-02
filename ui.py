import io
import json
import math
import subprocess
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="JBF: PRISM", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
        :root {
            color-scheme: dark;
        }
        .stApp {
            background: radial-gradient(circle at top left, #182848 0%, #111827 40%, #030712 100%);
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stStatusWidget"], [data-testid="collapsedControl"] {
            display: none !important;
        }
        .glass-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 24px;
            padding: 1.4rem;
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #f8fafc, #7dd3fc, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .stButton > button {
            border-radius: 999px;
            border: none;
            padding: 0.7rem 1.4rem;
            font-weight: 700;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            color: white;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 10px 24px rgba(59, 130, 246, 0.25);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 28px rgba(59, 130, 246, 0.35);
        }
        .stButton > button:disabled {
            background: #475569;
            color: #cbd5e1;
            box-shadow: none;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 0.7rem 0.9rem;
        }
        .status-line {
            color: #e2e8f0;
            margin: 0.2rem 0;
            font-size: 1rem;
        }
        .status-done {
            color: #4ade80;
            font-weight: 600;
        }
        .status-pending {
            color: #94a3b8;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent
JD_PATH = PROJECT_ROOT / "jd" / "job_description.txt"
DEFAULT_CANDIDATES_PATH = PROJECT_ROOT / "data" / "candidates.jsonl"
LARGE_CANDIDATES_PATHS = [
    Path(r"D:\JBF_Test\data\candidates.jsonl"),
    Path(r"D:\Documents\JBF_Test\data\candidates.jsonl"),
]
CANDIDATES_PATH = next((path for path in LARGE_CANDIDATES_PATHS if path.exists()), DEFAULT_CANDIDATES_PATH)

if "screen" not in st.session_state:
    st.session_state.screen = "landing"
if "job_file" not in st.session_state:
    st.session_state.job_file = None
if "candidate_file" not in st.session_state:
    st.session_state.candidate_file = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = []
if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None
if "pipeline_output" not in st.session_state:
    st.session_state.pipeline_output = ""
if "run_summary" not in st.session_state:
    st.session_state.run_summary = {}
if "pipeline_started" not in st.session_state:
    st.session_state.pipeline_started = False


def validate_upload(uploaded_file, expected_types):
    if uploaded_file is None:
        return False, "Please select a file."
    name = uploaded_file.name.lower()
    if not any(name.endswith(ext) for ext in expected_types):
        return False, "Unsupported File Format"
    try:
        if uploaded_file.getbuffer().nbytes == 0:
            return False, "Failed to Read File"
    except Exception:
        return False, "Failed to Read File"
    return True, "Uploaded successfully"


def _normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _parse_jsonish(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [{"name": text}]
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    return []


def _coerce_profile_value(row, aliases):
    for alias in aliases:
        if alias in row:
            value = row[alias]
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return value
    return ""


def _convert_candidate_rows(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        frame = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx"):
        frame = pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        payload = json.loads(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get("candidates", payload.get("data", []))
        raise ValueError("Unsupported JSON payload")
    elif name.endswith(".jsonl"):
        lines = [line for line in uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines() if line.strip()]
        return [json.loads(line) for line in lines]
    else:
        raise ValueError("Unsupported candidate dataset format")

    records = []
    for index, row in frame.iterrows():
        profile = {}
        profile["anonymized_name"] = _normalize_text(_coerce_profile_value(row, ["anonymized_name", "name", "candidate_name", "full_name"]))
        profile["headline"] = _normalize_text(_coerce_profile_value(row, ["headline", "headline_text", "title"]))
        profile["summary"] = _normalize_text(_coerce_profile_value(row, ["summary", "bio", "about"]))
        profile["location"] = _normalize_text(_coerce_profile_value(row, ["location", "city", "address"]))
        profile["country"] = _normalize_text(_coerce_profile_value(row, ["country", "nation"]))
        profile["years_of_experience"] = _normalize_text(_coerce_profile_value(row, ["years_of_experience", "experience", "years_experience"]))
        profile["current_title"] = _normalize_text(_coerce_profile_value(row, ["current_title", "current_role", "role"]))
        profile["current_company"] = _normalize_text(_coerce_profile_value(row, ["current_company", "company", "employer"]))
        profile["current_company_size"] = _normalize_text(_coerce_profile_value(row, ["current_company_size", "company_size"]))
        profile["current_industry"] = _normalize_text(_coerce_profile_value(row, ["current_industry", "industry"]))

        if "profile" in frame.columns and isinstance(row.get("profile"), dict):
            profile.update(row.get("profile"))

        candidate = {
            "candidate_id": _normalize_text(_coerce_profile_value(row, ["candidate_id", "id", "Candidate ID", "candidateId"])) or f"CAND_{index + 1:07d}",
            "profile": profile,
            "skills": _parse_jsonish(row.get("skills") if "skills" in frame.columns else None),
            "career_history": [],
            "education": [],
            "certifications": [],
            "languages": [],
            "redrob_signals": {},
        }
        records.append(candidate)

    return records


def _extract_docx_text(file_bytes):
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for node in root.findall(".//w:t", ns):
        text = node.text or ""
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _extract_pdf_text(file_bytes):
    try:
        import pypdf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PDF extraction requires the pypdf package.") from exc

    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def prepare_job_description(uploaded_file):
    if uploaded_file is None:
        raise ValueError("A Job Description file is required.")

    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if name.endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")
    elif name.endswith(".docx"):
        text = _extract_docx_text(file_bytes)
    elif name.endswith(".pdf"):
        text = _extract_pdf_text(file_bytes)
    else:
        raise ValueError("Unsupported job description format")

    JD_PATH.write_text(text, encoding="utf-8")
    return text


def prepare_candidate_dataset(uploaded_file):
    if uploaded_file is None:
        raise ValueError("A candidate dataset file is required.")

    records = _convert_candidate_rows(uploaded_file)
    CANDIDATES_PATH.parent.mkdir(exist_ok=True)
    with open(CANDIDATES_PATH, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return len(records)


def run_backend_pipeline():
    command = [sys.executable, "main.py"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
    return completed.returncode, output


def render_landing():
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:2.4rem 2rem; margin-top:1rem;">
            <div class="hero-title">JBF PRISM : Judgment Beyond Filters </div>
            <div style="font-size:1.55rem; font-weight:700; color:#f8fafc; margin:0.6rem 0 0.8rem 0;">
                “ Every Rank Has A Reason.”
            </div>
            <p style="color:#cbd5e1; max-width:720px; margin:0 auto 1rem auto; font-size:1.02rem;">
                Upload a Job Description and Candidate Dataset. The system will automatically perform candidate selection and return the best candidates.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start", use_container_width=True):
        st.session_state.screen = "upload"
        st.session_state.processing = False
        st.session_state.pipeline_started = False
        st.session_state.results_ready = False
        st.session_state.pipeline_error = None
        st.session_state.pipeline_output = ""
        st.rerun()

    st.write("")
    if st.button("View Sample Results", use_container_width=True):
        st.session_state.screen = "results"
        st.session_state.processing = False
        st.session_state.pipeline_started = False
        st.session_state.results_ready = True
        st.session_state.pipeline_error = None
        st.session_state.pipeline_output = ""
        st.rerun()


def render_upload():
    st.markdown("<div class='glass-card'><h2 style='margin-bottom:0.2rem;'>Upload Inputs</h2><p style='color:#cbd5e1;'>Add both required files to begin the analysis workflow.</p></div>", unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'><h4>Job Description</h4><p style='color:#cbd5e1;'>Accepted formats: .docx, .pdf, .txt</p></div>", unsafe_allow_html=True)
        uploaded_job = st.file_uploader("Job Description", type=["docx", "pdf", "txt"], label_visibility="collapsed")
        if uploaded_job is not None:
            valid, message = validate_upload(uploaded_job, ["docx", "pdf", "txt"])
            if valid:
                st.session_state.job_file = uploaded_job
                st.success("✓ Job Description Uploaded Successfully")
            else:
                st.error(f"✕ {message}")

    with col2:
        st.markdown("<div class='glass-card'><h4>Candidate Dataset</h4><p style='color:#cbd5e1;'>Accepted formats: .csv, .xlsx, .json, .jsonl</p></div>", unsafe_allow_html=True)
        uploaded_candidates = st.file_uploader("Candidate Dataset", type=["csv", "xlsx", "json", "jsonl"], label_visibility="collapsed")
        if uploaded_candidates is not None:
            valid, message = validate_upload(uploaded_candidates, ["csv", "xlsx", "json", "jsonl"])
            if valid:
                st.session_state.candidate_file = uploaded_candidates
                st.success("✓ Candidate Dataset Uploaded Successfully")
            else:
                st.error(f"✕ {message}")

    st.write("")

    can_start = st.session_state.job_file is not None and st.session_state.candidate_file is not None
    if can_start:
        st.success("Both files are ready. You can begin the analysis.")
    else:
        st.info("Please upload both files to enable the analysis button.")

    if st.button("Start Analysis", disabled=not can_start, use_container_width=True):
        st.session_state.screen = "processing"
        st.session_state.processing = True
        st.session_state.pipeline_started = False
        st.session_state.results_ready = False
        st.session_state.pipeline_error = None
        st.session_state.pipeline_output = ""
        st.session_state.pipeline_steps = []
        st.rerun()


def render_processing():
    st.markdown("<div class='glass-card'><h2 style='margin-bottom:0.2rem;'>Processing Pipeline</h2><p style='color:#cbd5e1;'>The analysis is running. Please wait while the platform executes the existing backend pipeline.</p></div>", unsafe_allow_html=True)
    st.write("")

    if st.session_state.pipeline_started:
        if st.session_state.pipeline_error:
            st.error(st.session_state.pipeline_error)
        elif st.session_state.results_ready:
            st.success("Results are ready.")
        if st.button("Back to Upload", use_container_width=True):
            st.session_state.screen = "upload"
            st.session_state.results_ready = False
            st.session_state.pipeline_error = None
            st.session_state.pipeline_output = ""
            st.session_state.pipeline_started = False
            st.rerun()
        return

    st.session_state.pipeline_started = True

    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    steps = [
        "Preparing uploaded files...",
        "Starting backend pipeline...",
        "Collecting generated outputs...",
    ]

    for idx, step in enumerate(steps):
        status_placeholder.markdown(f"<div class='status-line'>{step}</div>", unsafe_allow_html=True)
        progress_bar.progress((idx + 1) / len(steps))
        time.sleep(0.4)

    try:
        prepare_job_description(st.session_state.job_file)
        prepare_candidate_dataset(st.session_state.candidate_file)
        return_code, output = run_backend_pipeline()
        st.session_state.pipeline_output = output
        if return_code != 0:
            raise RuntimeError(output or "The backend pipeline exited with a non-zero status.")
        st.session_state.pipeline_error = None
        st.session_state.results_ready = True
        st.session_state.screen = "results"
        st.session_state.processing = False

        if (PROJECT_ROOT / "outputs" / "run_summary.json").exists():
            st.session_state.run_summary = json.loads((PROJECT_ROOT / "outputs" / "run_summary.json").read_text(encoding="utf-8"))
        st.rerun()
    except Exception as exc:  # pragma: no cover - runtime path
        st.session_state.pipeline_error = str(exc)
        st.session_state.results_ready = False
        st.session_state.screen = "upload"
        st.session_state.processing = False
        st.rerun()


def render_results():
    st.markdown("<div class='glass-card'><h2 style='margin-bottom:0.2rem;'>Results Dashboard</h2><p style='color:#cbd5e1;'>The ranked candidates and generated outputs are ready.</p></div>", unsafe_allow_html=True)
    st.write("")

    ranked_path = PROJECT_ROOT / "outputs" / "ranked_candidates.csv"
    diagnostics_path = PROJECT_ROOT / "outputs" / "diagnostics.json"
    run_summary_path = PROJECT_ROOT / "outputs" / "run_summary.json"
    submission_path = PROJECT_ROOT / "outputs" / "submission.csv"

    if not ranked_path.exists():
        st.warning("The backend pipeline did not produce the expected output files yet.")
        return

    ranked_df = pd.read_csv(ranked_path)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.exists() else {}
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8")) if run_summary_path.exists() else {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates Processed", diagnostics.get("processed_candidates", "n/a"))
    col2.metric("Candidates Retrieved", len(ranked_df))
    col3.metric("Top Candidate Score", f"{diagnostics.get('highest_score', 0):.2f}" if diagnostics else "n/a")
    col4.metric("Pipeline Runtime", f"{run_summary.get('total_runtime_seconds', 0):.2f}s")

    st.write("")
    st.dataframe(ranked_df, use_container_width=True, height=320)

    st.write("")
    st.markdown("<div class='glass-card'><h4>Generated Files</h4></div>", unsafe_allow_html=True)
    download_col1, download_col2, download_col3, download_col4 = st.columns(4)
    with download_col1:
        st.download_button("Download Submission CSV", data=submission_path.read_bytes(), file_name="submission.csv", mime="text/csv")
    with download_col2:
        st.download_button("Download Ranked Candidates", data=ranked_path.read_bytes(), file_name="ranked_candidates.csv", mime="text/csv")
    with download_col3:
        st.download_button("Download Diagnostics", data=diagnostics_path.read_bytes(), file_name="diagnostics.json", mime="application/json")
    with download_col4:
        st.download_button("Download Run Summary", data=run_summary_path.read_bytes(), file_name="run_summary.json", mime="application/json")

    if st.session_state.pipeline_output:
        st.write("")
        with st.expander("Pipeline log"):
            st.code(st.session_state.pipeline_output, language="text")

    st.write("")
    if st.button("Run New Analysis", use_container_width=True):
        st.session_state.screen = "upload"
        st.session_state.processing = False
        st.session_state.pipeline_started = False
        st.session_state.results_ready = False
        st.session_state.pipeline_error = None
        st.session_state.pipeline_output = ""
        st.rerun()


def main():
    st.markdown("<div style='position:fixed; inset:0; pointer-events:none; background:radial-gradient(circle at 20% 20%, rgba(125,211,252,0.12), transparent 25%), radial-gradient(circle at 80% 0%, rgba(167,139,250,0.12), transparent 25%);'></div>", unsafe_allow_html=True)
    if st.session_state.screen == "landing":
        render_landing()
    elif st.session_state.screen == "upload":
        render_upload()
    elif st.session_state.screen == "processing":
        render_processing()
    else:
        render_results()


if __name__ == "__main__":
    main()
