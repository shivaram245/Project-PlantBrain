import streamlit as st
import os
import time
import base64
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util

# --- Page Config (DEFINED ONLY ONCE) ---
st.set_page_config(page_title="Project PlantBrain", page_icon="🏭", layout="wide")

st.title("🏭 Project PlantBrain: Industrial Knowledge Intelligence Engine")
st.caption("ET AI Hackathon 2026 Submission | Fully Real Offline Edge RAG Architecture")
st.markdown("---")

# =====================================================================
# PLACE THE BASE64 BACKGROUND CODE RIGHT HERE (BELOW THE LINE)
# =====================================================================
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

local_image_path = os.path.join("static", "background.png")
img_base64 = get_base64_image(local_image_path)

if img_base64:
    st.markdown(
        f"""
        <style>
        html, body, .stApp, 
        [data-testid="stAppViewContainer"], 
        [data-testid="stMainViewContainer"], 
        [data-testid="stCanvas"],
        .main {{
            background-image: linear-gradient(rgba(15, 15, 20, 0.85), rgba(15, 15, 20, 0.92)), 
                              url("data:image/png;base64,{img_base64}") !important;
            background-attachment: fixed !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
        }}
        [data-testid="stAppViewBlockContainer"], .main .block-container {{
            background-color: transparent !important;
        }}
        [data-testid="stHeader"] {{
            background-color: rgba(0,0,0,0) !important;
        }}
        div[data-testid="stNotification"], .stInfo, .stWarning, .stSuccess {{
            background-color: rgba(25, 25, 35, 0.65) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }}
        h1, h2, h3, h4, p, span, label, .stMarkdown {{
            color: #ffffff !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.error("⚠️ Background Error: Python could not find 'static/background.png'")
# =====================================================================

# --- Initialize Local Embedding Model ---
def load_local_embedding_engine():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("Loading Local Semantic Embedding Engine Matrix into System Cache..."):
    local_ai_model = load_local_embedding_engine()

# --- Document Processing Core Engines ---
def parse_and_chunk_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    extracted_chunks = []
    for page_idx, page in enumerate(pdf_reader.pages):
        # SKIPS PAGE 1 (Index 0) so the cover page title doesn't hijack the vector math matches
        if page_idx == 0:
            continue
            
        page_text = page.extract_text()
        if page_text:
            blocks = [b.strip() for b in page_text.split("\n\n") if len(b.strip()) > 30]
            for block in blocks:
                extracted_chunks.append({
                    "text": block,
                    "source": f"{uploaded_file.name} (Page {page_idx + 1})"
                })
    return extracted_chunks

def parse_and_chunk_logs(uploaded_file):
    """Real time textual stream log partition manager"""
    raw_text = uploaded_file.read().decode("utf-8")
    log_lines = [line.strip() for line in raw_text.split("\n") if len(line.strip()) > 30]
    return [{"text": line, "source": uploaded_file.name} for line in log_lines]

# --- Initialize Session State Vars for Dynamic Sidebar ---
if "latency_val" not in st.session_state:
    st.session_state.latency_val = "0.00 seconds"
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# --- UI Sidebar Layout ---
with st.sidebar:
    st.header("📥 Ingestion Layer (Heterogeneous Silos)")
    st.markdown("Upload real industrial asset files to test the system dynamically.")
    
    manual_file = st.file_uploader("1. Asset Technical Manual (PDF Spec)", type=["pdf"])
    log_file = st.file_uploader("2. Historical Maintenance Logs (TXT Database)", type=["txt"])
    
    st.markdown("---")
    st.subheader("📊 Architectural Analytics")
    st.metric(label="RAG Engine Strategy", value="Edge Vector Index")
    
    # These containers placeholder layouts auto-update reactively without requiring st.rerun()
    latency_metric = st.metric(label="Calculated Processing Latency", value=st.session_state.latency_val)
    chunks_metric = st.metric(label="Total Scanned Knowledge Chunks", value=str(st.session_state.chunk_count))

# --- Main Application Execution View ---
st.subheader("💬 Query Collective Plant Intelligence Space")
user_prompt = st.text_input(
    "Enter an operational error alert or engineering inquiry:",
    value="Suction side pressure drop at the intake strainer. What are the operational limits?"
)

if st.button("🧠 Run Real Vector Diagnosis Pipeline"):
    if not manual_file or not log_file:
        st.error("Missing Data Error: Please upload both a manual PDF and the maintenance log text file in the sidebar.")
    elif not user_prompt:
        st.warning("Input Error: Please provide an engineering search query.")
    else:
        with st.spinner("Processing documents, building vector matrices, and matching vectors..."):
            start_runtime = time.time()
            
            # Step 1: Real-time File Parsing
            structural_manual_chunks = parse_and_chunk_pdf(manual_file)
            chronological_log_chunks = parse_and_chunk_logs(log_file)
            
            total_knowledge_pool = structural_manual_chunks + chronological_log_chunks
            
            if not total_knowledge_pool:
                st.error("Parsing Error: Could not extract valid text.")
            else:
                # Step 2: Local Vector Conversion Matrix Execution
                document_corpus_strings = [chunk["text"] for chunk in total_knowledge_pool]
                
                query_vector = local_ai_model.encode(user_prompt, convert_to_tensor=True)
                document_vectors = local_ai_model.encode(document_corpus_strings, convert_to_tensor=True)
                
                # Step 3: Compute Similarity (Cosine Similarity Matrix)
                similarity_scores = util.cos_sim(query_vector, document_vectors)[0].cpu().numpy()
                top_ranked_indices = np.argsort(similarity_scores)[::-1]
                
                # Step 4: Multi-Route Segregation Logic with Smarter Fallbacks
                best_manual = None
                best_log = None
                
                # Look for highest matching manual entry
                for rank_idx in top_ranked_indices:
                    matched_chunk = total_knowledge_pool[rank_idx]
                    if "pdf" in matched_chunk["source"].lower():
                        best_manual = matched_chunk
                        break
                
                # Look for highest matching maintenance log entry
                for rank_idx in top_ranked_indices:
                    matched_chunk = total_knowledge_pool[rank_idx]
                    if "txt" in matched_chunk["source"].lower():
                        best_log = matched_chunk
                        break
                
                # Calculate True Hardware Runtime Latency Metrics
                total_latency = time.time() - start_runtime
                
                # Save data to session state variables quietly
                st.session_state.latency_val = f"{total_latency:.2f} seconds"
                st.session_state.chunk_count = len(total_knowledge_pool)
                
                # --- Dynamic Sidebar Metric Overwrite Injection ---
                latency_metric.metric(label="Calculated Processing Latency", value=st.session_state.latency_val)
                chunks_metric.metric(label="Total Scanned Knowledge Chunks", value=str(st.session_state.chunk_count))
                
                # --- Output Grid Render ---
                st.markdown("### 🤖 Plant Brain System Diagnostics Report")
                st.success(f"Execution complete. Scanned `{len(total_knowledge_pool)}` text chunks in `{total_latency:.4f} seconds`.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🛠️ 1. OEM Manual Technical Specification Rules")
                    if best_manual:
                        st.info(f"\"{best_manual['text']}\"")
                        st.caption(f"📍 Linked Entity Location: `{best_manual['source']}`")
                    else:
                        st.write("No relevant manual information found.")
                        
                with col2:
                    st.markdown("#### ⏳ 2. Historical Pattern Matching & Shift Log Registry")
                    if best_log:
                        st.warning(f"\"{best_log['text']}\"")
                        st.caption(f"📍 Linked Entity Location: `{best_log['source']}`")
                    else:
                        st.write("No matching historical log entry found.")