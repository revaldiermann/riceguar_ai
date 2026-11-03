from pathlib import Path
from threading import Lock
import html
import logging
import os
import time
import warnings

import av
import cv2
import streamlit as st
import torch
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from streamlit_autorefresh import st_autorefresh

from src.data.transforms import val_test_transforms
from src.models.efficientnet import build_model

st.set_page_config(page_title="RiceGuard AI | Plant Health Intelligence", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")
logging.getLogger("aioice").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)

CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
CLASSES = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SCAN_DURATION_SECONDS = 10.0
INFERENCE_EVERY_N_FRAMES = 8
MINIMUM_SAMPLES = 5
LOW_CONFIDENCE_THRESHOLD = 60.0

DISEASE_INFO = {
    "Bacterial leaf blight": {
        "name_id": "Hawar Daun Bakteri", "icon": "🌾",
        "description": "Gejala dapat berupa perubahan warna kekuningan hingga lesi memanjang pada daun. Pola gejala bisa bervariasi menurut kondisi tanaman dan tahap penyakit.",
        "advice": ["Periksa beberapa daun dan tanaman, bukan hanya satu titik.", "Jaga pemupukan dan pengairan tetap seimbang.", "Mintalah konfirmasi petugas pertanian sebelum memilih tindakan pengendalian."],
    },
    "Brown spot": {
        "name_id": "Bercak Cokelat", "icon": "🍂",
        "description": "Biasanya tampak bercak cokelat berbentuk bulat atau oval. Tampilan bercak dapat dipengaruhi umur daun, nutrisi, dan kondisi lingkungan.",
        "advice": ["Amati bentuk, ukuran, dan penyebaran bercak pada beberapa daun.", "Periksa kondisi kesuburan tanah dan kesehatan tanaman secara menyeluruh.", "Gunakan benih sehat dan konsultasikan gejala yang terus meluas."],
    },
    "Leaf smut": {
        "name_id": "Penyakit Leaf Smut", "icon": "🌿",
        "description": "Model mengenali pola visual yang berasosiasi dengan kelas Leaf smut pada data latih. Prediksi visual saja tidak memastikan penyebab penyakit.",
        "advice": ["Bandingkan gejala pada beberapa daun dan tanaman.", "Hindari mengambil keputusan pengendalian hanya dari satu foto.", "Mintalah pemeriksaan petugas pertanian jika gejala menyebar."],
    },
}

with st.sidebar:
    st.markdown("<div class='sidebar-brand'><div class='brand-mark'>🌾</div><div><div class='brand-name'>RiceGuard <span>AI</span></div><div class='brand-subtitle'>PLANT HEALTH INTELLIGENCE</div></div></div>", unsafe_allow_html=True)
    st.markdown("---")
    theme_choice = st.radio("Appearance", ["☀️ Botanical Light", "🌙 Forest Dark"], index=0, key="theme_mode_selector")
    is_dark = "Dark" in theme_choice
    st.markdown("---")
    st.markdown("#### System overview")
    st.markdown(f"<div class='sidebar-stat'><span>Model engine</span><strong>EfficientNet-B0</strong></div><div class='sidebar-stat'><span>Runtime device</span><strong>{str(DEVICE).upper()}</strong></div><div class='sidebar-stat'><span>Supported classes</span><strong>03 classes</strong></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("RiceGuard AI · Research prototype")
    st.caption("Hasil prediksi perlu diverifikasi di lapangan.")

if is_dark:
    theme_vars = "--bg-main:#0B1210;--bg-soft:#101B17;--bg-card:#14211B;--text-main:#ECF5EF;--text-muted:#A1B5A9;--border:#263A30;--accent:#6EE7A0;--accent-strong:#238653;--accent-soft:#173B2A;--shadow:0 16px 40px rgba(0,0,0,.20);"
else:
    theme_vars = "--bg-main:#F5F8F5;--bg-soft:#EDF3EE;--bg-card:#FFFFFF;--text-main:#172B22;--text-muted:#63756B;--border:#DCE7DF;--accent:#16804A;--accent-strong:#12663B;--accent-soft:#E5F5EA;--shadow:0 12px 32px rgba(21,62,39,.07);"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {{{theme_vars}}}
html,body,[class*="css"]{{font-family:'DM Sans',sans-serif;color:var(--text-main)}}
.stApp{{background:var(--bg-main)}} [data-testid="stHeader"]{{background:transparent}}
[data-testid="stSidebar"]{{background:var(--bg-soft);border-right:1px solid var(--border)}}
.block-container{{max-width:1420px;padding-top:1.6rem;padding-bottom:3rem}}
h1,h2,h3,h4{{font-family:'Manrope',sans-serif!important;color:var(--text-main)!important;letter-spacing:-.035em}}
h1{{font-size:clamp(2rem,3vw,3.05rem)!important;font-weight:800!important;line-height:1.1!important}}h2{{font-size:1.65rem!important;font-weight:800!important}}h3{{font-size:1.2rem!important;font-weight:750!important}}
p,li,label,.stMarkdown{{color:var(--text-main)}}[data-testid="stCaptionContainer"] p{{color:var(--text-muted)!important;font-size:.84rem}}
.sidebar-brand{{display:flex;align-items:center;gap:12px;padding:6px 0 10px}}.brand-mark{{display:flex;align-items:center;justify-content:center;width:46px;height:46px;border-radius:15px;background:var(--accent-soft);font-size:1.55rem}}.brand-name{{font-family:'Manrope',sans-serif;font-weight:800;font-size:1.22rem;letter-spacing:-.04em}}.brand-name span{{color:var(--accent)}}.brand-subtitle{{font-size:.61rem;font-weight:700;letter-spacing:.12em;color:var(--text-muted);margin-top:2px}}
.sidebar-stat{{display:flex;flex-direction:column;gap:3px;padding:10px 12px;margin:7px 0;background:var(--bg-card);border:1px solid var(--border);border-radius:12px}}.sidebar-stat span{{font-size:.76rem;color:var(--text-muted)}}.sidebar-stat strong{{font-size:.93rem;font-weight:700}}
.hero{{position:relative;overflow:hidden;padding:30px 32px;border:1px solid var(--border);border-radius:24px;background:linear-gradient(125deg,var(--bg-card),var(--bg-soft));box-shadow:var(--shadow);margin-bottom:24px}}.hero:after{{content:' ';position:absolute;width:250px;height:250px;border-radius:50%;right:-75px;top:-100px;background:var(--accent-soft);opacity:.7}}.hero-content{{position:relative;z-index:1}}.eyebrow{{display:inline-flex;padding:6px 11px;border-radius:999px;background:var(--accent-soft);color:var(--accent);font-size:.74rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase}}.hero p{{max-width:760px;color:var(--text-muted);font-size:1.02rem;line-height:1.7;margin:12px 0 0}}.hero-meta{{display:flex;flex-wrap:wrap;gap:9px;margin-top:18px}}.meta-chip{{padding:7px 11px;border:1px solid var(--border);background:var(--bg-card);border-radius:10px;font-size:.78rem;font-weight:700;color:var(--text-muted)}}
.section-subtitle{{color:var(--text-muted);margin:0 0 18px;font-size:.93rem;line-height:1.6}}.panel{{background:var(--bg-card);border:1px solid var(--border);border-radius:18px;padding:20px;box-shadow:var(--shadow)}}.panel-label{{font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:var(--text-muted);margin-bottom:8px}}.result-title{{font-family:'Manrope',sans-serif;font-weight:800;font-size:1.55rem;line-height:1.25;letter-spacing:-.04em;margin:4px 0 6px}}.result-subtitle,.small-muted{{color:var(--text-muted);font-size:.84rem}}.confidence-number{{font-family:'Manrope',sans-serif;font-weight:800;font-size:2.45rem;line-height:1.1;letter-spacing:-.05em;color:var(--accent)}}.confidence-track{{height:8px;border-radius:999px;overflow:hidden;background:var(--bg-soft);margin:12px 0 8px}}.confidence-fill{{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--accent-strong),var(--accent))}}.info-box{{background:var(--bg-soft);border:1px solid var(--border);padding:14px 15px;border-radius:13px;color:var(--text-muted);font-size:.88rem;line-height:1.65}}
.stButton>button{{min-height:46px;border-radius:12px;font-family:'DM Sans',sans-serif;font-weight:700;font-size:.93rem;border:1px solid var(--border);transition:all .18s ease}}.stButton>button[kind="primary"]{{background:var(--accent-strong);color:#fff!important;border:1px solid var(--accent-strong)}}.stButton>button[kind="primary"]:hover{{filter:brightness(1.08);box-shadow:0 8px 20px rgba(22,128,74,.18);transform:translateY(-1px)}}.stButton>button:not([kind="primary"]){{background:var(--bg-card);color:var(--text-main)}}.stButton>button:not([kind="primary"]):hover{{border-color:var(--accent);color:var(--accent)}}
div[data-baseweb="radio"]{{gap:8px}}div[data-baseweb="radio"] label{{padding:8px 12px;border:1px solid var(--border);border-radius:11px;background:var(--bg-card);font-weight:600}}div[data-testid="stFileUploader"] section{{background:var(--bg-soft);border:1px dashed var(--border);border-radius:14px}}div[data-testid="stMetric"]{{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:14px 16px}}[data-testid="stExpander"]{{border:1px solid var(--border);border-radius:12px}}.footer{{border-top:1px solid var(--border);padding-top:18px;margin-top:32px;color:var(--text-muted);font-size:.8rem}}
/* Glassmorphism layer: translucent surfaces adapt to both themes */
.stApp{{background:radial-gradient(ellipse at 12% 0%,var(--accent-soft),transparent 38%),var(--bg-main)}}
[data-testid="stSidebar"]{{background:color-mix(in srgb,var(--bg-soft) 78%,transparent);backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);border-right:1px solid color-mix(in srgb,var(--border) 72%,transparent)}}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{{gap:.65rem}}
.glass-nav{{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:13px 18px;margin:0 0 22px;border:1px solid color-mix(in srgb,var(--border) 78%,transparent);border-radius:18px;background:color-mix(in srgb,var(--bg-card) 68%,transparent);box-shadow:var(--shadow);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px)}}
.glass-nav-brand{{display:flex;align-items:center;gap:10px;font-family:'Manrope',sans-serif;font-size:1rem;font-weight:800;letter-spacing:-.03em}}.glass-nav-icon{{display:grid;place-items:center;width:36px;height:36px;border-radius:12px;background:var(--accent-soft);font-size:1.15rem}}.glass-nav-meta{{font-size:.76rem;color:var(--text-muted);font-weight:700;letter-spacing:.04em}}
.hero,.panel,.sidebar-stat,div[data-testid="stMetric"],div[data-testid="stFileUploader"] section,[data-testid="stExpander"]{{background:color-mix(in srgb,var(--bg-card) 76%,transparent);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);border-color:color-mix(in srgb,var(--border) 78%,transparent)}}
.hero{{background:linear-gradient(125deg,color-mix(in srgb,var(--bg-card) 82%,transparent),color-mix(in srgb,var(--bg-soft) 72%,transparent))}}
.meta-chip{{background:color-mix(in srgb,var(--bg-card) 62%,transparent);backdrop-filter:blur(10px)}}
div[data-baseweb="radio"] label{{background:color-mix(in srgb,var(--bg-card) 68%,transparent);backdrop-filter:blur(12px);transition:all .18s ease}}div[data-baseweb="radio"] label:has(input:checked){{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent),0 6px 20px color-mix(in srgb,var(--accent) 14%,transparent)}}
.stButton>button{{backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px)}}
[data-testid="stPopover"] button,[data-testid="stSelectbox"] div[data-baseweb="select"]>div{{background:color-mix(in srgb,var(--bg-card) 72%,transparent);border-color:var(--border);backdrop-filter:blur(14px)}}
.footer{{backdrop-filter:blur(12px)}}
#MainMenu,footer{{visibility:hidden}}@media(max-width:768px){{.block-container{{padding-left:1rem;padding-right:1rem}}.hero{{padding:22px 20px}}.hero p{{font-size:.93rem}}.confidence-number{{font-size:2rem}}.glass-nav{{padding:11px 13px}}.glass-nav-meta{{font-size:.68rem}}}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Memuat model RiceGuard AI...")
def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {CHECKPOINT_PATH.resolve()}")
    model = build_model(pretrained=False).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(image: Image.Image) -> dict:
    model = load_model()
    tensor = val_test_transforms(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].detach().cpu()
    index = int(probabilities.argmax().item())
    return {"class_name": CLASSES[index], "confidence": float(probabilities[index].item() * 100), "probabilities": {CLASSES[i]: float(probabilities[i].item() * 100) for i in range(len(CLASSES))}}


def render_results(results: dict, show_samples: bool = False):
    class_name = results["class_name"]
    confidence = results["confidence"]
    info = DISEASE_INFO[class_name]
    st.markdown(f"<div class='panel'><div class='panel-label'>Predicted class</div><div class='result-title'>{info['icon']} {html.escape(info['name_id'])}</div><div class='result-subtitle'>Model class: {html.escape(class_name)}</div><div style='height:18px'></div><div class='panel-label'>Model confidence</div><div class='confidence-number'>{confidence:.2f}%</div><div class='confidence-track'><div class='confidence-fill' style='width:{max(0,min(100,confidence)):.2f}%'></div></div><div class='small-muted'>Confidence adalah skor probabilitas model, bukan jaminan diagnosis.</div></div>", unsafe_allow_html=True)
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning("Skor model relatif rendah. Coba foto yang lebih tajam, pencahayaan merata, dan daun memenuhi area gambar.")
    st.markdown("#### Ringkasan gejala")
    st.markdown(f"<div class='info-box'>{html.escape(info['description'])}</div>", unsafe_allow_html=True)
    st.markdown("#### Langkah pemeriksaan yang disarankan")
    for item in info["advice"]:
        st.markdown(f"- {item}")
    with st.expander("Lihat probabilitas semua kelas"):
        for name, probability in results["probabilities"].items():
            st.write(f"**{name}** · {probability:.2f}%")
            st.progress(min(max(probability / 100.0, 0.0), 1.0))
    if show_samples and "samples" in results:
        st.caption(f"Frame yang dipakai untuk agregasi: {results['samples']}")
    st.caption("RiceGuard AI adalah prototipe klasifikasi tiga kelas. Model tidak menandai lokasi bercak dan hasil perlu diverifikasi oleh petugas pertanian atau ahli tanaman.")


def build_rtc_configuration():
    # STUN helps establish peer-to-peer paths. TURN is optional for strict NAT/firewall networks.
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
    turn_url = os.getenv("RICEGUARD_TURN_URL", "").strip()
    turn_username = os.getenv("RICEGUARD_TURN_USERNAME", "").strip()
    turn_credential = os.getenv("RICEGUARD_TURN_CREDENTIAL", "").strip()
    if turn_url and turn_username and turn_credential:
        ice_servers.append({"urls": [u.strip() for u in turn_url.split(",") if u.strip()], "username": turn_username, "credential": turn_credential})
    return {"iceServers": ice_servers}


class LiveLeafProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.lock = Lock()
        self.started_at = None
        self.frame_count = 0
        self.probability_history = []
        self.finished = False
        self.final_result = None
        self.status = "Menunggu kamera..."
        self.last_error = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]
        x1, x2 = int(width * 0.25), int(width * 0.75)
        y1, y2 = int(height * 0.25), int(height * 0.75)
        roi = image[y1:y2, x1:x2]
        now = time.monotonic()
        with self.lock:
            if self.started_at is None:
                self.started_at = now
                self.status = "Memindai daun..."
            elapsed = now - self.started_at
            finished = self.finished
        if not finished and elapsed < SCAN_DURATION_SECONDS:
            self.frame_count += 1
            if self.frame_count % INFERENCE_EVERY_N_FRAMES == 0:
                try:
                    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    tensor = val_test_transforms(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
                    with torch.inference_mode():
                        probabilities = torch.softmax(self.model(tensor), dim=1)[0].detach().cpu()
                    with self.lock:
                        if not self.finished:
                            self.probability_history.append(probabilities)
                            self.last_error = None
                except Exception as exc:
                    with self.lock:
                        self.last_error = str(exc)
        with self.lock:
            elapsed = time.monotonic() - self.started_at if self.started_at is not None else 0.0
            if not self.finished and elapsed >= SCAN_DURATION_SECONDS:
                samples = list(self.probability_history)
                if len(samples) >= MINIMUM_SAMPLES:
                    average = torch.stack(samples).mean(dim=0)
                    index = int(average.argmax().item())
                    self.final_result = {"class_name": CLASSES[index], "confidence": float(average[index].item() * 100), "probabilities": {CLASSES[i]: float(average[i].item() * 100) for i in range(len(CLASSES))}, "samples": len(samples)}
                    self.status = "Pemindaian selesai"
                else:
                    self.final_result = None
                    self.status = "Sampel tidak cukup"
                self.finished = True
            finished, result = self.finished, self.final_result
            remaining = max(0.0, SCAN_DURATION_SECONDS - elapsed)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (12, 24, 18), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.43, image, 0.57, 0, image)
        color = (90, 225, 145) if not finished else (45, 190, 245)
        length = max(18, min(width, height) // 18)
        for p1, p2 in [((x1,y1),(x1+length,y1)),((x1,y1),(x1,y1+length)),((x2,y1),(x2-length,y1)),((x2,y1),(x2,y1+length)),((x1,y2),(x1+length,y2)),((x1,y2),(x1,y2-length)),((x2,y2),(x2-length,y2)),((x2,y2),(x2,y2-length))]:
            cv2.line(image, p1, p2, color, 3)
        if not finished:
            scan_y = int(y1 + ((elapsed * 0.5) % 1.0) * (y2 - y1))
            cv2.line(image, (x1 + 5, scan_y), (x2 - 5, scan_y), (90, 225, 145), 2)
            label = f"SCANNING  {remaining:04.1f}s"
        elif result:
            label = f"RESULT LOCKED  {result['confidence']:.1f}%"
        else:
            label = "SCAN INCOMPLETE - TRY AGAIN"
        cv2.rectangle(image, (x1, max(0, y1 - 38)), (x2, y1), (12, 24, 18), -1)
        cv2.putText(image, label[:55], (x1 + 9, max(23, y1 - 13)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (242, 248, 244), 1, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


st.markdown("<div class='glass-nav'><div class='glass-nav-brand'><span class='glass-nav-icon'>🌾</span><span>RiceGuard <span style='color:var(--accent)'>AI</span></span></div><div class='glass-nav-meta'>PLANT HEALTH INTELLIGENCE&nbsp; · &nbsp;EFFICIENTNET-B0</div></div>", unsafe_allow_html=True)

st.markdown("<div class='hero'><div class='hero-content'><div class='eyebrow'>✦ AI-powered crop observation</div><h1 style='margin:15px 0 0;'>Kenali gejala daun padi<br>lebih terstruktur.</h1><p>Analisis citra daun menggunakan EfficientNet-B0 untuk membantu pemeriksaan awal. Unggah foto atau gunakan kamera langsung dalam antarmuka yang dirancang untuk fokus, jelas, dan mudah digunakan.</p><div class='hero-meta'><span class='meta-chip'>🌿 3 kelas penyakit</span><span class='meta-chip'>⚡ PyTorch inference</span><span class='meta-chip'>📷 Upload & live camera</span></div></div></div>", unsafe_allow_html=True)
mode = st.radio("Pilih metode pemeriksaan", ["📁 Analisis foto", "📷 Kamera langsung"], horizontal=True, key="app_mode")

if mode == "📁 Analisis foto":
    st.markdown("## Analisis foto daun")
    st.markdown("<p class='section-subtitle'>Unggah foto yang fokus pada daun. Untuk hasil konsisten, gunakan cahaya merata dan hindari gerakan kamera.</p>", unsafe_allow_html=True)
    col_upload, col_results = st.columns([1.02, 0.98], gap="large")
    with col_upload:
        st.markdown("<div class='panel-label'>01 / Input specimen</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Pilih gambar dari perangkat", type=["jpg", "jpeg", "png"], key="riceguard_file_uploader", help="Format yang didukung: JPG, JPEG, PNG.")
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Pratinjau foto", use_container_width=True)
                if st.button("✨ Analisis gambar", type="primary", use_container_width=True, key="run_upload_prediction"):
                    with st.spinner("Model sedang menganalisis gambar..."):
                        st.session_state["upload_result"] = predict_image(image)
            except Exception as exc:
                st.error(f"Gambar tidak dapat diproses: {exc}")
    with col_results:
        st.markdown("<div class='panel-label'>02 / Analysis report</div>", unsafe_allow_html=True)
        result = st.session_state.get("upload_result")
        if result:
            render_results(result)
            if st.button("↻ Hapus hasil dan analisis foto lain", use_container_width=True):
                st.session_state.pop("upload_result", None)
                st.rerun()
        else:
            st.markdown("<div class='panel' style='min-height:230px;text-align:center;padding-top:55px'><div style='font-size:2.3rem'>🔬</div><div style='font-family:Manrope,sans-serif;font-weight:800;font-size:1.1rem;margin-top:12px'>Laporan analisis siap</div><div class='small-muted' style='max-width:320px;margin:8px auto;line-height:1.7'>Hasil kelas, confidence, ringkasan gejala, dan probabilitas akan muncul di area ini setelah gambar dianalisis.</div></div>", unsafe_allow_html=True)

elif mode == "📷 Kamera langsung":
    st.markdown("## Kamera observasi")
    st.markdown("<p class='section-subtitle'>Tempatkan satu daun di dalam area bidik. Sistem mengambil sampel selama 10 detik, lalu mengunci hasil berdasarkan rata-rata probabilitas.</p>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'><strong>Tips sebelum memulai</strong><br>• Izinkan akses kamera di browser.<br>• Posisikan daun di tengah kotak bidik dan tahan kamera agar stabil.<br>• Gunakan pencahayaan cukup dan latar belakang yang tidak terlalu ramai.</div>", unsafe_allow_html=True)
    if "camera_scan_id" not in st.session_state:
        st.session_state["camera_scan_id"] = 0
    scan_id = st.session_state["camera_scan_id"]
    try:
        ctx = webrtc_streamer(key=f"riceguard-live-{scan_id}", mode=WebRtcMode.SENDRECV, video_processor_factory=LiveLeafProcessor, media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False}, rtc_configuration=build_rtc_configuration(), async_processing=True)
    except Exception as exc:
        st.error(f"Tidak dapat memulai komponen kamera: {exc}")
        ctx = None
    if ctx is not None and ctx.state.playing:
        st_autorefresh(interval=700, limit=None, key=f"riceguard-refresh-{scan_id}")
    processor = ctx.video_processor if ctx is not None else None
    if processor is not None:
        with processor.lock:
            result = processor.final_result
            finished = processor.finished
            status = processor.status
            error = processor.last_error
            sample_count = len(processor.probability_history)
        st.markdown(f"**Status pemindaian:** `{status}`")
        st.progress(min(sample_count / MINIMUM_SAMPLES, 1.0), text=f"Sampel terkumpul: {sample_count}/{MINIMUM_SAMPLES} minimum")
        if error:
            st.warning(f"Frame terakhir gagal diproses: {error}")
        if not finished:
            st.info("Pemindaian berjalan. Pertahankan daun di area bidik sampai hasil terkunci.")
        elif result is not None:
            st.success("Pemindaian selesai — hasil akhir sudah dikunci.")
            render_results(result, show_samples=True)
            if st.button("↻ Pindai spesimen berikutnya", type="primary", use_container_width=True):
                st.session_state["camera_scan_id"] += 1
                st.rerun()
        else:
            st.warning("Sampel belum cukup untuk menghasilkan prediksi. Pastikan kamera stabil, daun terlihat jelas, lalu ulangi pemindaian.")
            if st.button("Coba pemindaian lagi", type="primary", use_container_width=True):
                st.session_state["camera_scan_id"] += 1
                st.rerun()
    elif ctx is not None:
        st.markdown("<div class='panel' style='margin-top:16px'><div class='panel-label'>Camera connection</div><div style='font-family:Manrope,sans-serif;font-weight:800;font-size:1.05rem'>Menunggu koneksi kamera</div><p class='small-muted' style='line-height:1.7;margin-bottom:0'>Klik <strong>START</strong> pada komponen kamera. Jika koneksi terus menunggu, ikuti langkah pemecahan masalah di bawah.</p></div>", unsafe_allow_html=True)
    with st.expander("Troubleshooting: koneksi kamera terlalu lama"):
        st.markdown("""
**Jika muncul “Connection is taking longer than expected”:**

1. Jika berjalan di komputer sendiri, buka `http://localhost:8501`. Browser biasanya mengizinkan kamera pada localhost atau HTTPS.
2. Pastikan izin Camera pada browser diatur ke **Allow**, lalu muat ulang halaman.
3. Tutup aplikasi lain yang memakai kamera dan coba browser Chrome/Edge terbaru.
4. Coba jaringan lain atau matikan VPN sementara. Firewall, NAT kampus/kantor, atau jaringan seluler dapat menghalangi WebRTC.
5. Konfigurasi STUN publik sudah disertakan. STUN membantu menemukan jalur koneksi, tetapi tidak menjamin koneksi di semua jaringan.
6. Untuk deployment publik atau jaringan ketat, siapkan server **TURN** yang bisa dijangkau browser dan server aplikasi. Isi kredensial melalui environment variables, jangan menuliskannya di kode.
""")
        st.code("export RICEGUARD_TURN_URL='turn:your-turn-host:3478'\nexport RICEGUARD_TURN_USERNAME='your-username'\nexport RICEGUARD_TURN_CREDENTIAL='your-password'\npython -m streamlit run app.py", language="bash")
        st.caption("Ganti nilai contoh dengan kredensial TURN milik Anda. Jangan unggah kredensial TURN ke GitHub.")

st.markdown("<div class='footer'><strong>RiceGuard AI</strong> · EfficientNet-B0 · PyTorch · Streamlit WebRTC<br><br>Prototipe pendukung pemeriksaan awal. Hasil model bukan diagnosis agronomi definitif.</div>", unsafe_allow_html=True)
