import io
import os
import re
import json
import glob
import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from urllib.parse import urlparse
from collections import Counter
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image, ImageFilter, ImageOps
import imagehash

# ==== CLIP : import sécurisé ====
try:
    import torch
    import open_clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

# ==== SHEETS : import sécurisé ====
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_SHEETS = True
except ImportError:
    HAS_SHEETS = False

# ===================================================
# CONFIG
# ===================================================

BASE_DOMAIN = "https://www.planete-sfactory.com"

STOPWORDS_FR = frozenset({
    "a", "à", "afin", "ai", "aie", "ainsi", "après", "as", "au", "aucun", "aucune",
    "aujourd", "aujourd'hui", "auquel", "aussi", "autre", "autres", "aux",
    "auxquelles", "auxquels", "avec", "avoir", "avant", "bon", "car", "ce", "ceci",
    "cela", "celle", "celles", "celui", "cependant", "ces", "cet", "cette", "ceux",
    "chacun", "chaque", "chez", "ci", "comme", "comment", "d", "d'", "dans", "de",
    "des", "du", "dedans", "dehors", "depuis", "devant", "doit", "donc", "dont",
    "dos", "droite", "déjà", "elle", "elles", "en", "encore", "enfin", "entre",
    "envers", "et", "etc", "été", "être", "eu", "eux", "fait", "faites", "fois",
    "font", "furent", "haut", "hors", "ici", "il", "ils", "je", "jusqu", "jusque",
    "l", "l'", "la", "le", "les", "leur", "leurs", "lors", "lorsque", "lui", "là",
    "ma", "maintenant", "mais", "me", "même", "mes", "moi", "moins", "mon", "ne",
    "ni", "non", "nos", "notre", "nous", "nouveau", "nouveaux", "on", "ont", "ou",
    "où", "par", "parce", "parfois", "parmi", "pas", "peu", "peut", "plus",
    "plusieurs", "pour", "pourquoi", "près", "qu", "qu'", "quand", "que", "quel",
    "quelle", "quelles", "quels", "qui", "quoi", "rarement", "s", "s'", "sa",
    "sans", "se", "sera", "seront", "ses", "si", "sien", "sienne", "siennes",
    "siens", "sinon", "soi", "soit", "sommes", "son", "sont", "souvent", "sous",
    "sur", "t", "ta", "tandis", "te", "tes", "toi", "ton", "toujours", "tous",
    "tout", "toute", "toutes", "très", "tu", "un", "une", "vers", "voici", "voilà",
    "vos", "votre", "vous", "vu", "y", "ça"
})

# ===================================================
# STREAMLIT CONFIG & CUSTOM CSS
# ===================================================

st.set_page_config(
    page_title="S-Factory Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pour un design moderne
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    /* Variables de thème */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --surface: #1e1e2e;
        --surface-light: #2a2a3e;
        --text: #e2e8f0;
        --text-muted: #94a3b8;
        --border: #3f3f5a;
        --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    /* Base */
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header personnalisé */
    .main-header {
        background: var(--gradient-1);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Cards */
    .stat-card {
        background: linear-gradient(145deg, #2a2a3e 0%, #1e1e2e 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.2);
        border-color: var(--primary);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: var(--text-muted);
        font-size: 0.9rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress container */
    .progress-container {
        background: var(--surface-light);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border);
    }
    
    .progress-title {
        font-weight: 600;
        color: var(--text);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 500;
        gap: 0.5rem;
    }
    
    .status-running {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16162a 100%);
    }
    
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background: var(--surface-light);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text);
    }
    
    section[data-testid="stSidebar"] .stTextArea > div > div > textarea {
        background: var(--surface-light);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text);
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--gradient-1);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--surface-light);
        border-radius: 10px;
        border: 1px solid var(--border);
        font-weight: 600;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: var(--surface-light);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Anomaly level badges */
    .anomaly-ok { color: #10b981; font-weight: 600; }
    .anomaly-faible { color: #f59e0b; font-weight: 600; }
    .anomaly-moyenne { color: #f97316; font-weight: 600; }
    .anomaly-forte { color: #ef4444; font-weight: 600; }
    
    /* Animation keyframes */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--surface);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }
    section[data-testid="stSidebar"] * {
    color: white !important;
    }
    section[data-testid="stSidebar"] label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

if "results_by_category" not in st.session_state:
    st.session_state.results_by_category = []

# ===================================================
# OPTIMIZED HTTP SESSION
# ===================================================

_session_lock = threading.Lock()
_http_session = None

def get_http_session() -> requests.Session:
    """Session HTTP réutilisable avec retry et connection pooling."""
    global _http_session
    if _http_session is None:
        with _session_lock:
            if _http_session is None:
                session = requests.Session()
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
                adapter = HTTPAdapter(
                    max_retries=retry_strategy,
                    pool_connections=20,
                    pool_maxsize=20
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                session.headers.update({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })
                _http_session = session
    return _http_session

# ===================================================
# CLIP STATE
# ===================================================

CLIP_MODEL = None
CLIP_PREPROCESS = None
CLIP_DEVICE = "cpu"

def init_clip():
    global CLIP_MODEL, CLIP_PREPROCESS, CLIP_DEVICE
    if not HAS_CLIP or CLIP_MODEL is not None:
        return
    CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.to(CLIP_DEVICE)
    model.eval()
    CLIP_MODEL = model
    CLIP_PREPROCESS = preprocess

# ===================================================
# SHEETS HELPERS
# ===================================================

SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def sheets_client_from_streamlit_secrets():
    sa_info = st.secrets.get("gcp_service_account", None)
    if not sa_info:
        raise RuntimeError("Secret manquant: st.secrets['gcp_service_account']")
    creds = Credentials.from_service_account_info(sa_info, scopes=SHEETS_SCOPES)
    return gspread.authorize(creds)

def sheets_client_from_service_account_file(sa_json_path: str):
    creds = Credentials.from_service_account_file(sa_json_path, scopes=SHEETS_SCOPES)
    return gspread.authorize(creds)

def extract_sheet_id(sheet_url_or_id: str) -> str:
    s = (sheet_url_or_id or "").strip()
    if not s:
        return ""
    if "docs.google.com" in s and "/spreadsheets/d/" in s:
        return s.split("/spreadsheets/d/")[1].split("/")[0]
    return s

def sanitize_sheet_title(title: str, max_len: int = 90) -> str:
    t = re.sub(r"[\[\]\:\*\?\/\\]", " ", title).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return (t or "Categorie")[:max_len]

def ensure_worksheet(sh, title: str, rows: int = 2000, cols: int = 50):
    try:
        return sh.worksheet(title)
    except Exception:
        return sh.add_worksheet(title=title, rows=rows, cols=cols)

def write_df_to_sheet(sh, title: str, df: pd.DataFrame):
    ws = ensure_worksheet(sh, title, rows=max(2000, len(df) + 50), cols=max(50, len(df.columns) + 10))
    ws.clear()
    if df.empty:
        ws.update("A1", [["(vide)"]])
        return
    payload = [df.columns.astype(str).tolist()] + df.astype(str).fillna("").values.tolist()
    ws.update(payload, value_input_option="USER_ENTERED")

def read_sheet_as_df(sh, title: str) -> pd.DataFrame:
    try:
        ws = sh.worksheet(title)
        values = ws.get_all_values()
        if not values or not values[0]:
            return pd.DataFrame()
        return pd.DataFrame(values[1:], columns=values[0])
    except Exception:
        return pd.DataFrame()

def ensure_exclusions_sheet(sh):
    title = "EXCLUSIONS"
    ws = ensure_worksheet(sh, title, rows=2000, cols=10)
    values = ws.get_all_values()
    if not values:
        ws.update("A1:E1", [["category_url", "product_url", "nom", "decision_client", "date"]])
    return ws

def load_excluded_urls(sh, category_url: str) -> set:
    df_excl = read_sheet_as_df(sh, "EXCLUSIONS")
    if df_excl.empty or "category_url" not in df_excl.columns or "product_url" not in df_excl.columns:
        return set()
    df_excl = df_excl[df_excl["category_url"].astype(str) == str(category_url)]
    return set(u for u in df_excl["product_url"].astype(str).tolist() if u and u != "nan")

def update_exclusions_from_df(sh, category_url: str, df: pd.DataFrame):
    if df.empty or "decision_client" not in df.columns:
        return
    ws = ensure_exclusions_sheet(sh)
    excl = df[df["decision_client"].astype(str).str.strip().str.lower().isin(
        ["a_exclure", "exclure", "exclude", "a exclure"]
    )].copy()
    if excl.empty:
        return
    existing = read_sheet_as_df(sh, "EXCLUSIONS")
    existing_urls = set(existing.get("product_url", pd.Series([], dtype=str)).astype(str).tolist()) if not existing.empty else set()
    excl = excl[["url", "nom", "decision_client"]].rename(columns={"url": "product_url"})
    excl.insert(0, "category_url", category_url)
    excl["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    excl = excl[~excl["product_url"].astype(str).isin(existing_urls)]
    if not excl.empty:
        ws.append_rows(excl.astype(str).values.tolist(), value_input_option="USER_ENTERED")

def load_decisions_map_from_all_products(sh) -> dict:
    df = read_sheet_as_df(sh, "ALL_PRODUCTS")
    if df.empty or "url" not in df.columns or "decision_client" not in df.columns:
        return {}
    return {u.strip(): d.strip() for u, d in zip(df["url"].astype(str), df["decision_client"].astype(str)) if u.strip()}

def delete_all_tabs_except(sh, keep_titles: set):
    for t in keep_titles:
        ensure_worksheet(sh, t)
    for ws in sh.worksheets():
        if ws.title not in keep_titles:
            try:
                sh.del_worksheet(ws)
            except Exception:
                pass

# ===================================================
# UTILS HTTP / HTML
# ===================================================

def absolutize_url(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return BASE_DOMAIN.rstrip("/") + "/" + href.lstrip("/")

@lru_cache(maxsize=500)
def get_soup_cached(url: str) -> str:
    """Cache le HTML brut pour éviter les requêtes répétées."""
    session = get_http_session()
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text

def get_soup(url: str) -> BeautifulSoup:
    html = get_soup_cached(url)
    return BeautifulSoup(html, "html.parser") # lxml est plus rapide que html.parser

# ===================================================
# PAGINATION
# ===================================================

def find_next_page_url(soup: BeautifulSoup, current_url: str) -> Optional[str]:
    link = soup.find("link", rel=lambda x: x and "next" in x)
    if link and link.get("href"):
        return absolutize_url(link["href"])
    
    a = soup.find("a", rel=lambda x: x and "next" in x, href=True)
    if a and a.get("href"):
        return absolutize_url(a["href"])
    
    a = soup.select_one("a.next[href], li.next a[href], .pages a.next[href], .pagination a.next[href]")
    if a and a.get("href"):
        return absolutize_url(a["href"])
    
    for a in soup.find_all("a", href=True):
        txt = a.get_text(" ", strip=True).lower()
        if txt in ("suivant", "suivante", "next", ">", "›"):
            return absolutize_url(a["href"])
    return None

# ===================================================
# CATEGORIE
# ===================================================

def get_category_slug(category_url: str) -> str:
    path = urlparse(category_url).path
    segments = [s for s in path.split("/") if s]
    if not segments:
        return ""
    last = segments[-1]
    if last.endswith(".html") and len(segments) >= 2:
        return segments[-2]
    return last

def extract_category_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else ""

def extract_category_intro_text(soup: BeautifulSoup, max_chars: int = 4000) -> str:
    selectors = [
        ".category-description", "#category-description",
        ".category-description.std", ".std",
        ".category-view .category-description",
    ]
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            txt = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
            if len(txt) > 50:
                return txt[:max_chars]
    
    main = soup.select_one("main") or soup.select_one("#main") or soup.select_one(".main") or soup.body
    if not main:
        return ""
    
    for kill_sel in [".category-products", ".products-grid", ".products", "ol.products", "ul.products-grid"]:
        for k in main.select(kill_sel):
            k.decompose()
    
    parts = []
    for p in main.find_all(["p", "li"], limit=30):
        t = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        if t and len(t) > 40:
            parts.append(t)
        if sum(len(x) for x in parts) > max_chars:
            break
    return " ".join(parts)[:max_chars]

def build_category_main_term(category_title: str, category_url: str) -> str:
    raw = (category_title or "").lower().strip()
    slug = get_category_slug(category_url).replace("-", " ").lower().strip()
    candidate_text = raw if len(raw) > 2 else slug
    candidate_text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9\s-]", " ", candidate_text)
    candidate_text = re.sub(r"\s+", " ", candidate_text).strip()
    tokens = [t for t in candidate_text.split(" ") if t and t not in STOPWORDS_FR and len(t) > 3]
    
    for preferred in ["vaporisateur", "cigarette", "bang", "grinder", "pipe", "briquet", "papier", "filtre", "tube"]:
        if preferred in candidate_text:
            return preferred
    return tokens[0] if tokens else "produit"

# ===================================================
# GET PRODUCT URLS (OPTIMIZED)
# ===================================================

def get_product_urls(category_url: str, max_pages: int = 80) -> list:
    urls = set()
    visited_pages = set()
    current = category_url

    # Chemin de catégorie pour filtrer les produits hors catégorie
    cat_path = urlparse(category_url).path.rstrip("/") + "/"

    for _ in range(max_pages):
        if current in visited_pages:
            break
        visited_pages.add(current)

        soup = get_soup(current)

        selectors = [
            "a.product-item-link[href]",
            "h2.product-name a[href]",
            ".product-name a[href]",
            "a.product-image[href]",
        ]

        page_urls = set()

        def accept_href(href: str) -> bool:
            if not href:
                return False
            absu = absolutize_url(href)
            p = urlparse(absu).path
            # on garde uniquement les .html qui sont dans l'arborescence de la catégorie
            return p.endswith(".html") and (p.startswith(cat_path) or cat_path in p)

        for sel in selectors:
            for a in soup.select(sel):
                href = a.get("href", "")
                if accept_href(href):
                    page_urls.add(absolutize_url(href))

        # Fallback MAIS filtré (sinon tu repêches n'importe quoi)
        if not page_urls:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if accept_href(href):
                    page_urls.add(absolutize_url(href))

        urls.update(page_urls)

        next_url = find_next_page_url(soup, current)
        if not next_url:
            break
        current = next_url

    return sorted(urls)


# ===================================================
# SCRAPING PRODUIT
# ===================================================

SKU_REGEX = re.compile(r"\b([A-Z]{3,5}-\d{4})\b")

def extract_sku(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        zone = h1.parent.get_text(" ", strip=True) if h1.parent else h1.get_text(" ", strip=True)
        m = SKU_REGEX.search(zone)
        if m:
            return m.group(1)
    txt = soup.get_text(" ", strip=True)
    m = SKU_REGEX.search(txt)
    return m.group(1) if m else ""

def extract_caracteristiques(soup: BeautifulSoup) -> list:
    for tag in soup.find_all(["h2", "h3"]):
        if "Caractéristiques" in tag.get_text(strip=True):
            ul = tag.find_next("ul")
            if ul:
                return [li.get_text(strip=True) for li in ul.find_all("li")]
    return []

def extract_price(soup: BeautifulSoup) -> str:
    special = soup.find("p", class_="special-price")
    if special:
        span = special.find("span", class_="price")
        if span:
            return span.get_text(strip=True)
    span = soup.find("span", class_="price")
    return span.get_text(strip=True) if span else ""

def extract_image_url(soup: BeautifulSoup) -> str:
    for attr in ["data-src", "data-original"]:
        img = soup.find("img", attrs={attr: True})
        if img:
            url = img.get(attr)
            if url:
                return absolutize_url(url)
    
    product_imgs = soup.select("img[src*='catalog'], img[src*='product'], img[src*='media']")
    for img in product_imgs:
        src = img.get("src")
        if src and len(src) > 10:
            return absolutize_url(src)
    
    img = soup.find("img")
    return absolutize_url(img.get("src")) if img and img.get("src") else ""

def extract_tech_specs(soup: BeautifulSoup) -> dict:
    for tag in soup.find_all(["h2", "h3"]):
        text = tag.get_text(strip=True).upper()
        if "INFORMATIONS" in text and "COMPL" in text:
            table = tag.find_next("table")
            if table:
                specs = {}
                for tr in table.find_all("tr"):
                    cells = tr.find_all(["th", "td"])
                    if len(cells) >= 2:
                        key = cells[0].get_text(strip=True)
                        val = cells[1].get_text(strip=True)
                        if key:
                            specs[key] = val
                return specs
    return {}

def extract_description_block(soup: BeautifulSoup) -> Tuple[str, str, List[str]]:
    for tag in soup.find_all(["h2", "h3"]):
        if "Description" in tag.get_text(strip=True):
            nodes = []
            for sib in tag.next_siblings:
                name = getattr(sib, "name", None)
                if name in ("h2", "h3"):
                    break
                if name is not None:
                    nodes.append(sib)
            
            if not nodes:
                return "", "", []
            
            description_html = "".join(str(n) for n in nodes)
            parts = [n.get_text(" ", strip=True) for n in nodes if hasattr(n, "get_text")]
            description_txt = " ".join(p for p in parts if p)
            
            desc_img_urls = []
            seen = set()
            for n in nodes:
                for img in n.find_all("img"):
                    src = img.get("src") or img.get("data-src") or img.get("data-original")
                    if src:
                        url = absolutize_url(src)
                        if url not in seen:
                            seen.add(url)
                            desc_img_urls.append(url)
            
            return description_txt, description_html, desc_img_urls
    return "", "", []

def check_image_url_ok(url: str) -> Tuple[bool, str]:
    if not url or not isinstance(url, str):
        return False, "url vide"
    try:
        session = get_http_session()
        r = session.get(url, timeout=10, stream=True)
        if r.status_code >= 400:
            return False, f"http {r.status_code}"
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "image" not in ctype:
            return False, f"content-type={ctype or 'inconnu'}"
        return True, "ok"
    except Exception as e:
        return False, f"erreur: {e}"

def title_qty_size_signals(title: str) -> dict:
    t = (title or "").strip()
    qty = None
    m = re.search(r"(?i)\b[x×]\s?(\d{1,4})\b", t)
    if m:
        qty = int(m.group(1))
    else:
        m = re.search(r"(?i)\b(lot|pack)\s*(de)?\s*(\d{1,4})\b", t)
        if m:
            qty = int(m.group(3))
    
    size_raw, size_val, size_unit = "", None, ""
    m = re.search(r"(?i)\b(\d+(?:[.,]\d+)?)\s?(cm|mm|m|ml|l|g|kg)\b", t)
    if m:
        size_raw = m.group(0)
        size_val = float(m.group(1).replace(",", "."))
        size_unit = m.group(2).lower()
    
    return {
        "title_qty": qty if qty is not None else "",
        "title_size_raw": size_raw,
        "title_size_value": size_val if size_val is not None else "",
        "title_size_unit": size_unit,
    }

def scrape_product(url: str) -> dict:
    """Scrape un produit - optimisé."""
    soup = get_soup(url)
    
    title_tag = soup.find("h1")
    name = title_tag.get_text(strip=True) if title_tag else ""
    sku = extract_sku(soup)
    
    brand = ""
    for p in soup.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if txt.startswith("Marque :"):
            brand = txt.replace("Marque :", "").strip()
            break
    
    price = extract_price(soup)
    description, description_html, desc_img_urls = extract_description_block(soup)
    caracteristiques_list = extract_caracteristiques(soup)
    caracteristiques = " | ".join(caracteristiques_list)
    image_url = extract_image_url(soup)
    
    # Vérification images description en parallèle
    desc_img_ok, desc_img_reason, desc_img_sim_to_main = [], [], []
    main_img = download_pil_image(image_url) if image_url else None
    main_ph = imagehash.phash(main_img) if main_img is not None else None
    
    for u in desc_img_urls:
        ok, reason = check_image_url_ok(u)
        desc_img_ok.append(ok)
        desc_img_reason.append(reason)
        
        sim = ""
        if ok and main_ph is not None:
            img2 = download_pil_image(u)
            if img2 is not None:
                try:
                    ph2 = imagehash.phash(img2)
                    sim = round(phash_similarity(main_ph, ph2), 3)
                except Exception:
                    pass
        desc_img_sim_to_main.append(sim)
    
    signals = title_qty_size_signals(name)
    tech_specs = extract_tech_specs(soup)
    
    data = {
        "url": url,
        "nom": name,
        "sku": sku,
        "marque": brand,
        "prix": price,
        "caracteristiques": caracteristiques,
        "description": description,
        "description_html": description_html,
        "desc_img_urls": " | ".join(desc_img_urls),
        "desc_img_ok": " | ".join([str(x) for x in desc_img_ok]),
        "desc_img_reason": " | ".join(desc_img_reason),
        "desc_img_sim_to_main": " | ".join([str(x) for x in desc_img_sim_to_main]),
        "image_url": image_url,
        **signals,
    }
    
    for k, v in tech_specs.items():
        col_name = "fiche_" + re.sub(r"[éèê]", "e", re.sub(r"[àâ]", "a", k.strip().lower().replace(" ", "_").replace("'", "_")))
        data[col_name] = v
    
    return data

def scrape_products_parallel(urls: List[str], max_workers: int = 8, progress_callback=None) -> List[dict]:
    """Scrape plusieurs produits en parallèle."""
    results = [None] * len(urls)
    completed = 0
    
    def scrape_with_index(idx_url):
        idx, url = idx_url
        try:
            return idx, scrape_product(url)
        except Exception as e:
            return idx, {
                "url": url, "nom": "ERREUR", "sku": "", "marque": "", "prix": "",
                "caracteristiques": f"Erreur: {e}", "description": "", "description_html": "",
                "desc_img_urls": "", "desc_img_ok": "", "desc_img_reason": "",
                "desc_img_sim_to_main": "", "image_url": "", "title_qty": "",
                "title_size_raw": "", "title_size_value": "", "title_size_unit": "",
            }
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scrape_with_index, (i, url)): i for i, url in enumerate(urls)}
        
        for future in as_completed(futures):
            idx, product = future.result()
            results[idx] = product
            completed += 1
            
            if progress_callback:
                progress_callback(completed, len(urls), urls[idx])
    
    return results

# ===================================================
# ENRICHISSEMENT
# ===================================================

def price_to_float(price_str):
    if not isinstance(price_str, str):
        return np.nan
    txt = price_str.replace("€", "").replace(" ", "").replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return np.nan

def extract_materiaux(carac: str) -> str:
    if not isinstance(carac, str):
        return ""
    for part in carac.split("|"):
        if "Matériaux" in part or "Matériau" in part:
            return part.split(":", 1)[-1].strip()
    return ""

def enrich_structured_features(df: pd.DataFrame, category_main_term: str) -> pd.DataFrame:
    if df.empty:
        return df
    
    df["prix_num"] = df["prix"].apply(price_to_float)
    
    txt_all = (
        df.get("nom", "").fillna("") + " " +
        df.get("description", "").fillna("") + " " +
        df.get("caracteristiques", "").fillna("")
    ).str.lower()
    
    term = (category_main_term or "produit").strip().lower()
    
    df["has_category_word"] = txt_all.str.contains(re.escape(term))
    df["has_vaporisateur_word"] = txt_all.str.contains("vaporisateur")
    df["is_portable"] = txt_all.str.contains("portable")
    df["is_salon"] = txt_all.str.contains("de salon|salon")
    df["chauffe_briquet"] = txt_all.str.contains("briquet|torche")
    df["chauffe_batterie"] = txt_all.str.contains("batterie|rechargeable|accu|usb-c|usb c")
    df["has_capsule_word"] = txt_all.str.contains("capsule|capsules|doseuse")
    df["has_accessoire_word"] = txt_all.str.contains(
        "accessoire|adaptateur|embout|bouchon|joint|grille|piece detachee|pieces detachees"
    )
    df["materiaux"] = df["caracteristiques"].apply(extract_materiaux)
    df["description_len"] = df["description"].fillna("").str.len()
    
    fiche_cols = [c for c in df.columns if c.startswith("fiche_")]
    df["nb_fiche_champs"] = len(fiche_cols)
    if fiche_cols:
        df["nb_fiche_non_vides"] = df[fiche_cols].apply(
            lambda row: sum(bool(str(v).strip()) for v in row), axis=1
        )
        df["taux_fiche_completude"] = df["nb_fiche_non_vides"] / df["nb_fiche_champs"]
    else:
        df["nb_fiche_non_vides"] = 0
        df["taux_fiche_completude"] = np.nan
    
    return df

# ===================================================
# FICHE TECHNIQUE COHERENCE
# ===================================================

def _norm_txt(v) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = re.sub(r"\s+", " ", s).replace("'", "'")
    return s

def _tech_field_consensus(series: pd.Series, min_ratio: float = 0.6) -> Tuple[str, float]:
    vals = [_norm_txt(x) for x in series.tolist()]
    vals = [v for v in vals if v and v != "nan"]
    if not vals:
        return ("", 0.0)
    c = Counter(vals)
    top_val, top_cnt = c.most_common(1)[0]
    ratio = top_cnt / max(1, len(vals))
    return (top_val, ratio) if ratio >= min_ratio else ("", ratio)

def compute_fiche_technique_incoherence(df: pd.DataFrame, consensus_ratio: float = 0.6) -> pd.DataFrame:
    if df.empty:
        df["score_fiche_technique"] = 0
        df["ano_fiche_technique"] = False
        df["fiche_tech_issues"] = ""
        df["nb_fiche_champs_consideres"] = 0
        return df
    
    fiche_cols = [c for c in df.columns if c.startswith("fiche_") and "dimension" not in c]
    if not fiche_cols:
        df["score_fiche_technique"] = 0
        df["ano_fiche_technique"] = False
        df["fiche_tech_issues"] = ""
        df["nb_fiche_champs_consideres"] = 0
        return df
    
    stable_cols = []
    consensus = {}
    for col in fiche_cols:
        top_val, ratio = _tech_field_consensus(df[col], min_ratio=consensus_ratio)
        if top_val:
            stable_cols.append(col)
            consensus[col] = (top_val, ratio)
    
    issues_list, scores = [], []
    for _, row in df.iterrows():
        issues, penalty = [], 0
        for col in stable_cols:
            v = _norm_txt(row.get(col, ""))
            if v and v != "nan":
                top_val, _ = consensus[col]
                if v != top_val:
                    penalty += 12
                    issues.append(f"{col.replace('fiche_', '')}: '{v}' ≠ '{top_val}'")
        
        scores.append(int(min(100, penalty)))
        issues_list.append(" ; ".join(issues))
    
    df["score_fiche_technique"] = scores
    df["ano_fiche_technique"] = df["score_fiche_technique"] >= 25
    df["fiche_tech_issues"] = issues_list
    df["nb_fiche_champs_consideres"] = len(stable_cols)
    
    # Rareté sur colonne "type"
    type_col = next((c for c in fiche_cols if c.lower() == "fiche_type"), 
                    next((c for c in fiche_cols if "type" in c.lower()), ""))
    
    if type_col:
        vals = df[type_col].fillna("").astype(str).apply(_norm_txt)
        vals = vals[(vals != "") & (vals != "nan")]
        if len(vals) > 0:
            c = Counter(vals.tolist())
            total = sum(c.values())
            rare_vals = {k for k, v in c.items() if (v / max(1, total)) <= 0.10 and v >= 1}
            
            new_scores, new_issues = [], []
            for s, iss, v in zip(df["score_fiche_technique"].tolist(), df["fiche_tech_issues"].tolist(), 
                                  df[type_col].fillna("").astype(str).tolist()):
                v_norm = _norm_txt(v)
                add_pen, add_txt = 0, ""
                if v_norm and v_norm in rare_vals:
                    add_pen = 25
                    add_txt = f"{type_col.replace('fiche_', '')}: valeur rare '{v_norm}'"
                ns = int(min(100, (s or 0) + add_pen))
                ni = (iss + " ; " + add_txt) if add_txt and iss else (add_txt or iss)
                new_scores.append(ns)
                new_issues.append(ni)
            
            df["score_fiche_technique"] = new_scores
            df["fiche_tech_issues"] = new_issues
            df["ano_fiche_technique"] = df["score_fiche_technique"] >= 25
    
    return df

# ===================================================
# MOTS-CLES
# ===================================================

def tokenize_fr(text: str) -> list:
    text = (text or "").lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [t for t in text.split(" ") if t and t not in STOPWORDS_FR and len(t) > 3]

def compute_common_title_terms(df: pd.DataFrame, top_k: int = 12) -> list:
    titles = df.get("nom", pd.Series([], dtype=str)).fillna("").astype(str).tolist()
    counter = Counter()
    for t in titles:
        counter.update(tokenize_fr(t))
    return [w for (w, _) in counter.most_common(top_k)]

def description_keyword_coverage(desc: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    toks = set(tokenize_fr(desc))
    return sum(1 for k in keywords if k in toks) / max(1, len(keywords))

# ===================================================
# SIMILARITÉ TEXTE
# ===================================================

def compute_similarity(df: pd.DataFrame, category_intro_text: str) -> pd.DataFrame:
    if df.empty:
        df["similarite_categorie"] = 1.0
        df["mots_cles_principaux"] = ""
        return df
    
    product_texts = (
        df.get("nom", "").fillna("").astype(str) + " " +
        df.get("description", "").fillna("").astype(str)
    )
    
    cat_text = (category_intro_text or "").strip()
    all_texts = pd.concat([product_texts, pd.Series([cat_text])], ignore_index=True)
    
    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS_FR), ngram_range=(1, 2), min_df=1)
    X_all = vectorizer.fit_transform(all_texts)
    X_prod = X_all[:-1]
    X_cat = X_all[-1]
    
    sim_cat = cosine_similarity(X_prod, X_cat)
    df["similarite_categorie"] = np.asarray(sim_cat).ravel()
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    main_terms = []
    for i in range(X_prod.shape[0]):
        row = X_prod[i].toarray()[0]
        if (row > 0).sum() == 0:
            main_terms.append("")
        else:
            top_idx = row.argsort()[-7:][::-1]
            terms = [t for t in feature_names[top_idx] if len(t) > 2]
            main_terms.append(", ".join(terms[:5]))
    
    df["mots_cles_principaux"] = main_terms
    return df

# ===================================================
# IMAGE SIMILARITY (OPTIMIZED)
# ===================================================

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    n = np.linalg.norm(vec)
    return vec / n if n != 0 else vec

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return np.nan
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na != 0 and nb != 0 else np.nan

def download_pil_image(url: str) -> Optional[Image.Image]:
    try:
        session = get_http_session()
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None

def crop_to_object(img: Image.Image, white_thresh: int = 245, margin: float = 0.10) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    non_white = np.any(arr < white_thresh, axis=2)
    if not np.any(non_white):
        return img
    
    ys, xs = np.where(non_white)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    
    h, w = arr.shape[:2]
    dy = int((y1 - y0 + 1) * margin)
    dx = int((x1 - x0 + 1) * margin)
    
    y0, y1 = max(0, y0 - dy), min(h - 1, y1 + dy)
    x0, x1 = max(0, x0 - dx), min(w - 1, x1 + dx)
    
    return img.crop((x0, y0, x1 + 1, y1 + 1))

def prepare_image_for_analysis(img: Image.Image) -> Image.Image:
    return crop_to_object(img, white_thresh=245, margin=0.10).resize((256, 256))

def color_signature(img: Image.Image, bins_h=24, bins_s=8, bins_v=8) -> np.ndarray:
    img = prepare_image_for_analysis(img)
    hsv = img.convert("HSV")
    arr = np.asarray(hsv, dtype=np.uint8)
    h_bin = (arr[:, :, 0].astype(np.int32) * bins_h) // 256
    s_bin = (arr[:, :, 1].astype(np.int32) * bins_s) // 256
    v_bin = (arr[:, :, 2].astype(np.int32) * bins_v) // 256
    
    hist = np.zeros((bins_h, bins_s, bins_v), dtype=np.float32)
    np.add.at(hist, (h_bin, s_bin, v_bin), 1.0)
    return _l2_normalize(hist.ravel())

def shape_signature(img: Image.Image, size=128) -> np.ndarray:
    img = prepare_image_for_analysis(img)
    gray = ImageOps.grayscale(img).resize((size, size))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.clip(np.asarray(edges, dtype=np.float32) * 1.5, 0, 255)
    hist, _ = np.histogram(arr.ravel(), bins=32, range=(0, 255))
    return _l2_normalize(hist.astype(np.float32))

def phash_similarity(h1, h2) -> float:
    if h1 is None or h2 is None:
        return np.nan
    return float(1 - ((h1 - h2) / 64))

def get_clip_embedding(url: str) -> Optional[np.ndarray]:
    if not HAS_CLIP:
        return None
    init_clip()
    try:
        session = get_http_session()
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img_t = CLIP_PREPROCESS(img).unsqueeze(0).to(CLIP_DEVICE)
        with torch.no_grad():
            emb = CLIP_MODEL.encode_image(img_t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]
    except Exception:
        return None

def process_image_features(url: str) -> Dict[str, Any]:
    """Traite une image et retourne tous ses features."""
    result = {"hash": None, "embedding": None, "shape": None, "color": None}
    img = download_pil_image(url) if isinstance(url, str) and url.strip() else None
    
    if img is None:
        return result
    
    try:
        result["hash"] = imagehash.phash(img)
    except Exception:
        pass
    
    try:
        result["shape"] = shape_signature(img)
    except Exception:
        pass
    
    try:
        result["color"] = color_signature(img)
    except Exception:
        pass
    
    if HAS_CLIP:
        result["embedding"] = get_clip_embedding(url)
    
    return result

def compute_image_similarity_optimized(df: pd.DataFrame, max_workers: int = 6) -> pd.DataFrame:
    """Version optimisée avec traitement parallèle des images."""
    if df.empty:
        for col in ["image_hash", "image_embedding", "similarite_image_moyenne", 
                    "similarite_forme_moyenne", "similarite_couleur_moyenne", "similarite_image_globale_moyenne"]:
            df[col] = np.nan if "similarite" in col else None
        return df
    
    urls = df["image_url"].tolist()
    features = []
    
    # Traitement parallèle des images
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        features = list(executor.map(process_image_features, urls))
    
    hashes = [f["hash"] for f in features]
    embeddings = [f["embedding"] for f in features]
    shape_vecs = [f["shape"] for f in features]
    color_vecs = [f["color"] for f in features]
    
    df["image_hash"] = [str(h) if h is not None else None for h in hashes]
    df["image_embedding"] = embeddings
    df["image_shape_vec"] = shape_vecs
    df["image_color_vec"] = color_vecs
    
    n = len(df)
    sim_img_mean, sim_shape_mean, sim_color_mean, sim_global_mean = [], [], [], []
    
    # Pré-calcul des matrices pour vectorisation
    for i in range(n):
        sims_main, sims_shape, sims_color, sims_global = [], [], [], []
        
        for j in range(n):
            if i == j:
                continue
            
            # Similarité principale (CLIP ou pHash)
            if HAS_CLIP and embeddings[i] is not None and embeddings[j] is not None:
                s_main = float(np.dot(embeddings[i], embeddings[j]))
                w_main = 0.55
            elif hashes[i] is not None and hashes[j] is not None:
                s_main = phash_similarity(hashes[i], hashes[j])
                w_main = 0.45
            else:
                s_main = np.nan
                w_main = 0
            
            s_sh = _cosine(shape_vecs[i], shape_vecs[j])
            s_col = _cosine(color_vecs[i], color_vecs[j])
            
            if not np.isnan(s_main):
                sims_main.append(s_main)
            if not np.isnan(s_sh):
                sims_shape.append(s_sh)
            if not np.isnan(s_col):
                sims_color.append(s_col)
            
            # Score global pondéré
            parts, weights = [], []
            if not np.isnan(s_main):
                parts.append(s_main)
                weights.append(w_main)
            if not np.isnan(s_sh):
                parts.append(s_sh)
                weights.append(0.25 if HAS_CLIP else 0.30)
            if not np.isnan(s_col):
                parts.append(s_col)
                weights.append(0.20 if HAS_CLIP else 0.25)
            
            if parts and sum(weights) > 0:
                sims_global.append(float(np.average(parts, weights=weights)))
        
        sim_img_mean.append(float(np.mean(sims_main)) if sims_main else np.nan)
        sim_shape_mean.append(float(np.mean(sims_shape)) if sims_shape else np.nan)
        sim_color_mean.append(float(np.mean(sims_color)) if sims_color else np.nan)
        sim_global_mean.append(float(np.mean(sims_global)) if sims_global else np.nan)
    
    df["similarite_image_moyenne"] = sim_img_mean
    df["similarite_forme_moyenne"] = sim_shape_mean
    df["similarite_couleur_moyenne"] = sim_color_mean
    df["similarite_image_globale_moyenne"] = sim_global_mean
    
    return df

# ===================================================
# FLAGS / SCORE
# ===================================================

def add_outlier_flags_and_reasons(df: pd.DataFrame, category_main_term: str) -> pd.DataFrame:
    if df.empty:
        for col in ["anomalie_score", "niveau_anomalie", "suspect", "raison_suspecte", 
                    "reco_action", "score_fiche_technique", "ano_fiche_technique",
                    "score_description", "ano_description"]:
            df[col] = 0 if "score" in col else (False if col in ["suspect", "ano_fiche_technique", "ano_description"] else "")
        return df
    
    threshold_img = float(np.nanquantile(df["similarite_image_moyenne"], 0.05)) if df["similarite_image_moyenne"].notna().any() else None
    threshold_cat = float(np.nanquantile(df["similarite_categorie"], 0.10)) if df["similarite_categorie"].notna().any() else None
    
    if "prix_num" in df.columns and df["prix_num"].notna().any():
        q1 = float(df["prix_num"].quantile(0.25))
        q3 = float(df["prix_num"].quantile(0.75))
    else:
        q1 = q3 = None
    
    majority_portable = df["is_portable"].mean() > 0.5 if "is_portable" in df.columns else False
    term = (category_main_term or "produit").strip()
    
    results = {"scores": [], "niveaux": [], "suspects": [], "reasons": [], "actions": [],
               "score_ft": [], "ano_ft": [], "score_desc": [], "ano_desc": []}
    
    for _, row in df.iterrows():
        row_reasons, row_actions = [], []
        score, score_ft, score_desc = 0, 0, 0
        
        # Similarité catégorie
        if threshold_cat is not None:
            simc = row.get("similarite_categorie", np.nan)
            if isinstance(simc, (int, float)) and not np.isnan(simc) and simc < threshold_cat:
                score += 20
                row_reasons.append("texte produit peu aligné avec l'introduction de la catégorie")
                # row_actions.append("Vérifier classement catégorie") - désactivé
        
        # Similarité image
        if threshold_img is not None:
            sim_img = row.get("similarite_image_moyenne", np.nan)
            if isinstance(sim_img, (int, float)) and not np.isnan(sim_img) and sim_img < threshold_img:
                score += 20
                row_reasons.append("image très différente des autres produits")
                row_actions.append("Vérifier cohérence image")
        
        # Images description
        desc_ok = str(row.get("desc_img_ok", "")).lower()
        if desc_ok and desc_ok != "nan":
            if any(x in ("false", "0") for x in desc_ok.split("|")):
                score += 15
                row_reasons.append("image(s) description cassée(s)")
                row_actions.append("Corriger balises <img>")
        
        try:
            sims = [float(s.strip()) for s in str(row.get("desc_img_sim_to_main", "")).split("|") 
                    if s.strip() and s.strip() not in ("nan", "")]
            if sims and min(sims) < 0.35:
                score += 10
                row_reasons.append("image(s) description potentiellement hors-sujet")
                row_actions.append("Vérifier images description")
        except Exception:
            pass
        
        title_qty = row.get("title_qty", "")
        if title_qty and isinstance(title_qty, (int, float)) and title_qty >= 10:
            score += 30
            row_reasons.append(f"produit vendu en lot/display ({int(title_qty)} unites)")
            row_actions.append("Verifier si ce lot doit etre dans une categorie 'grossiste' ou 'display'")
        # Flags produit - désactivés pour éviter les faux positifs
        # (capsule, accessoire, category_word)
        
        if majority_portable and row.get("is_salon", False):
            score += 15
            row_reasons.append("mentionne 'de salon' (catégorie majoritairement portable)")
            row_actions.append("Vérifier classification")
        
        # Fiche technique
        ft_score_raw = row.get("score_fiche_technique", 0)
        if isinstance(ft_score_raw, (int, float)) and not np.isnan(ft_score_raw) and ft_score_raw > 0:
            if ft_score_raw >= 30:
                score_ft += 30
                row_reasons.append("fiche technique incohérente")
                det = row.get("fiche_tech_issues", "")
                if det:
                    row_reasons.append(f"→ {det}")
                row_actions.append("Vérifier fiche technique")
            elif ft_score_raw >= 20:
                score_ft += 20
                row_reasons.append("fiche technique légèrement incohérente")
                row_actions.append("Contrôler fiche technique")
        
        # Description keywords - désactivé pour éviter faux positifs
        # cov = row.get("desc_kw_coverage", np.nan)
        # if isinstance(cov, (int, float)) and not np.isnan(cov) and cov < 0.20:
        #     score_desc += 15
        #     row_reasons.append("description peu alignée avec les mots courants")
        #     row_actions.append("Enrichir description")
        
        score += score_ft + score_desc
        
        # Prix comme confirmation
        if score >= 30 and q1 is not None:
            prix = row.get("prix_num")
            if isinstance(prix, (int, float)) and not np.isnan(prix) and (prix < q1 or prix > q3):
                score += 10
                row_reasons.append("prix atypique (renforce doute)")
                row_actions.append("Vérifier cohérence prix")
        
        score = min(score, 100)
        niveau = "OK" if score < 30 else ("faible" if score < 50 else ("moyenne" if score < 70 else "forte"))
        suspect = score >= 65
        
        results["scores"].append(score)
        results["niveaux"].append(niveau)
        results["suspects"].append(suspect)
        results["reasons"].append(" ; ".join(dict.fromkeys(r for r in row_reasons if r)) or "Aucune anomalie")
        results["actions"].append(" | ".join(dict.fromkeys(a for a in row_actions if a)) or "OK")
        results["score_ft"].append(int(score_ft))
        results["ano_ft"].append(bool(ft_score_raw >= 25) if isinstance(ft_score_raw, (int, float)) else False)
        results["score_desc"].append(int(score_desc))
        results["ano_desc"].append(bool(score_desc >= 15))
    
    df["anomalie_score"] = results["scores"]
    df["niveau_anomalie"] = results["niveaux"]
    df["suspect"] = results["suspects"]
    df["raison_suspecte"] = results["reasons"]
    df["reco_action"] = results["actions"]
    df["score_fiche_technique"] = results["score_ft"]
    df["ano_fiche_technique"] = results["ano_ft"]
    df["score_description"] = results["score_desc"]
    df["ano_description"] = results["ano_desc"]
    
    return df

# ===================================================
# CLIENT COLUMNS
# ===================================================

def add_client_decision_columns(df: pd.DataFrame, decisions_map: Optional[dict] = None) -> pd.DataFrame:
    if df.empty:
        df["decision_client"] = ""
        df["exclure_prochaine_analyse"] = False
        return df
    
    if "decision_client" not in df.columns:
        df["decision_client"] = ""
    
    if decisions_map:
        df["decision_client"] = df["url"].astype(str).apply(
            lambda u: (decisions_map.get(u.strip(), "") or "").strip()
        )
    
    df["exclure_prochaine_analyse"] = df["decision_client"].astype(str).str.strip().str.lower().isin(
        ["a_exclure", "exclure", "exclude", "a exclure"]
    )
    return df

# ===================================================
# UI COMPONENTS
# ===================================================

def render_header():
    st.markdown("""
    <div class="main-header slide-in">
        <h1>🔍 S-Factory Product Analyzer</h1>
        <p>Analyse de cohérence et détection d'anomalies pour vos catalogues produits</p>
    </div>
    """, unsafe_allow_html=True)

def render_stats_cards(total_products: int, total_suspects: int, categories_count: int, avg_score: float):
    cols = st.columns(4)
    
    stats = [
        (total_products, "Produits analysés", "📦"),
        (total_suspects, "Suspects détectés", "⚠️"),
        (categories_count, "Catégories", "📁"),
        (f"{avg_score:.1f}", "Score moyen", "📊"),
    ]
    
    for col, (value, label, icon) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

def render_progress_status(status_type: str, message: str, details: str = ""):
    status_classes = {
        "running": "status-running",
        "success": "status-success",
        "warning": "status-warning",
        "error": "status-error"
    }
    icons = {
        "running": "🔄",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    
    st.markdown(f"""
    <div class="progress-container slide-in">
        <div class="progress-title">
            <span class="status-badge {status_classes.get(status_type, 'status-running')}">
                {icons.get(status_type, '🔄')} {message}
            </span>
        </div>
        {f'<p style="color: var(--text-muted); margin: 0.5rem 0 0 0; font-size: 0.9rem;">{details}</p>' if details else ''}
    </div>
    """, unsafe_allow_html=True)

def render_anomaly_badge(niveau: str) -> str:
    colors = {
        "OK": "#10b981",
        "faible": "#f59e0b",
        "moyenne": "#f97316",
        "forte": "#ef4444"
    }
    return f'<span style="color: {colors.get(niveau, "#94a3b8")}; font-weight: 600;">{niveau}</span>'

# ===================================================
# MAIN UI
# ===================================================

render_header()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    urls_text = st.text_area(
        "📎 URLs catégories (1 par ligne)",
        value="https://www.planete-sfactory.com/vaporisateur/vaporisateur-portable/",
        height=140,
        help="Entrez les URLs des catégories à analyser"
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        analyse_image = st.checkbox("🖼️ Analyse images", value=True)
    with col2:
        parallel_scraping = st.checkbox("⚡ Mode parallèle", value=True)
    
    max_pages = st.slider("📄 Max pages/catégorie", 1, 200, 80)
    max_workers = st.slider("🔧 Workers parallèles", 2, 16, 8) if parallel_scraping else 1
    
    st.markdown("---")
    st.markdown("### 📊 Export Google Sheets")
    
    export_sheets = st.checkbox("Activer export Sheets", value=True)
    
    if export_sheets:
        sheet_url_or_id = st.text_input(
            "URL/ID du tableur",
            value="https://docs.google.com/spreadsheets/d/1lSqjJb-6HZQVfsDVrjeN2wULtFW7luK4d-HmBQOlx58/edit?gid=0#gid=0"
        )
        
        auth_mode = st.radio(
            "Mode d'authentification",
            ["Streamlit secrets", "Fichier JSON local"],
            index=0
        )
        
        sa_json_path = ""
        if auth_mode == "Fichier JSON local":
            sa_json_path = st.text_input("Chemin JSON service account")
        
        wipe_tabs = st.checkbox("🗑️ Supprimer anciens onglets", value=False)

# Main content
st.markdown("---")

run = st.button("🚀 Lancer l'analyse", use_container_width=True)

# Placeholders pour le suivi en temps réel
progress_placeholder = st.empty()
status_placeholder = st.empty()
metrics_placeholder = st.empty()

if run:
    category_urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    
    if not category_urls:
        st.error("⚠️ Veuillez entrer au moins une URL de catégorie.")
    else:
        st.session_state.results_by_category = []
        
        # Init Sheets
        spreadsheet_id = extract_sheet_id(sheet_url_or_id) if export_sheets else ""
        sh = None
        sheets_error = None
        decisions_map = {}
        
        if export_sheets:
            if not HAS_SHEETS:
                sheets_error = "Packages manquants: gspread + google-auth"
            elif not spreadsheet_id:
                sheets_error = "ID tableur invalide"
            else:
                try:
                    if auth_mode == "Streamlit secrets":
                        client = sheets_client_from_streamlit_secrets()
                    else:
                        if not sa_json_path or not os.path.exists(sa_json_path):
                            raise FileNotFoundError("JSON service account introuvable")
                        client = sheets_client_from_service_account_file(sa_json_path)
                    
                    sh = client.open_by_key(spreadsheet_id)
                    ensure_exclusions_sheet(sh)
                    
                    if wipe_tabs:
                        delete_all_tabs_except(sh, keep_titles={"EXCLUSIONS"})
                    
                    decisions_map = load_decisions_map_from_all_products(sh)
                except Exception as e:
                    sheets_error = str(e)
        
        # Progress
        progress_bar = progress_placeholder.progress(0)
        
        total_products_scraped = 0
        total_suspects_found = 0
        
        for idx_cat, category_url in enumerate(category_urls, start=1):
            with status_placeholder.container():
                render_progress_status(
                    "running",
                    f"Catégorie {idx_cat}/{len(category_urls)}",
                    category_url
                )
            
            try:
                # Lecture catégorie
                soup_cat = get_soup(category_url)
                category_title = extract_category_title(soup_cat)
                category_intro = extract_category_intro_text(soup_cat)
                category_main_term = build_category_main_term(category_title, category_url)
                
                # URLs produits
                product_urls = get_product_urls(category_url, max_pages=int(max_pages))
                
                # Exclusions
                if sh is not None:
                    try:
                        excluded = load_excluded_urls(sh, category_url)
                        if excluded:
                            product_urls = [u for u in product_urls if u not in excluded]
                    except Exception:
                        pass
                
                if not product_urls:
                    df = pd.DataFrame()
                else:
                    # Scraping
                    if parallel_scraping:
                        prod_progress = st.progress(0, text="Scraping produits...")
                        
                        def update_progress(completed, total, url):
                            prod_progress.progress(
                                completed / total,
                                text=f"Produit {completed}/{total}"
                            )
                        
                        results = scrape_products_parallel(product_urls, max_workers=max_workers, progress_callback=update_progress)
                        prod_progress.empty()
                    else:
                        results = []
                        for i, prod_url in enumerate(product_urls):
                            try:
                                results.append(scrape_product(prod_url))
                            except Exception as e:
                                results.append({
                                    "url": prod_url, "nom": "ERREUR", "sku": "", "marque": "",
                                    "prix": "", "caracteristiques": f"Erreur: {e}",
                                    "description": "", "description_html": "", "desc_img_urls": "",
                                    "desc_img_ok": "", "desc_img_reason": "", "desc_img_sim_to_main": "",
                                    "image_url": "", "title_qty": "", "title_size_raw": "",
                                    "title_size_value": "", "title_size_unit": "",
                                })
                    
                    df = pd.DataFrame(results)
                    
                    # Enrichissement
                    common_title_terms = compute_common_title_terms(df, top_k=12)
                    df["category_title_keywords"] = ", ".join(common_title_terms)
                    df["desc_kw_coverage"] = df["description"].fillna("").astype(str).apply(
                        lambda d: description_keyword_coverage(d, common_title_terms)
                    )
                    
                    df = enrich_structured_features(df, category_main_term)
                    df = compute_similarity(df, category_intro)
                    
                    if analyse_image:
                        with st.spinner("🖼️ Analyse des images..."):
                            df = compute_image_similarity_optimized(df, max_workers=max_workers)
                    
                    df = compute_fiche_technique_incoherence(df, consensus_ratio=0.6)
                    df = add_outlier_flags_and_reasons(df, category_main_term)
                    df = add_client_decision_columns(df, decisions_map=decisions_map)
                    
                    total_products_scraped += len(df)
                    total_suspects_found += df["suspect"].sum() if "suspect" in df.columns else 0
                
                st.session_state.results_by_category.append({
                    "category_url": category_url,
                    "category_title": category_title,
                    "category_main_term": category_main_term,
                    "category_intro": category_intro,
                    "df": df
                })
                
                progress_bar.progress(idx_cat / len(category_urls))
                
            except Exception as e:
                st.error(f"❌ Erreur catégorie {category_url}: {e}")
                st.session_state.results_by_category.append({
                    "category_url": category_url,
                    "category_title": "",
                    "category_main_term": "produit",
                    "category_intro": "",
                    "df": pd.DataFrame()
                })
                progress_bar.progress(idx_cat / len(category_urls))
        
        # Success
        with status_placeholder.container():
            render_progress_status("success", "Analyse terminée!", 
                                   f"{total_products_scraped} produits analysés, {total_suspects_found} suspects")
        
        # Export Sheets
        if export_sheets:
            if sheets_error:
                st.error(f"❌ Export Sheets impossible: {sheets_error}")
            else:
                try:
                    with st.spinner("📤 Export vers Google Sheets..."):
                        summary_rows, all_suspects_rows, all_products_rows = [], [], []
                        
                        for item in st.session_state.results_by_category:
                            cat_url = item["category_url"]
                            title = item["category_title"] or get_category_slug(cat_url) or "Categorie"
                            tab_title = sanitize_sheet_title(title)
                            
                            df_all = item["df"].copy()
                            if df_all.empty:
                                df_all = pd.DataFrame({"info": [f"Aucun produit pour: {cat_url}"]})
                            else:
                                df_all.insert(0, "category_url", cat_url)
                                df_all.insert(1, "category_title", item["category_title"])
                                df_all.insert(2, "mot_cle_principal", item["category_main_term"])
                                all_products_rows.append(df_all)
                            
                            write_df_to_sheet(sh, tab_title, df_all)
                            
                            if "suspect" in df_all.columns:
                                df_sus = df_all[df_all["suspect"] == True].copy()
                                if not df_sus.empty:
                                    all_suspects_rows.append(df_sus)
                            
                            n_total = len(df_all) if "url" in df_all.columns else 0
                            n_sus = int(df_all["suspect"].sum()) if "suspect" in df_all.columns else 0
                            n_excl = int(df_all["exclure_prochaine_analyse"].sum()) if "exclure_prochaine_analyse" in df_all.columns else 0
                            
                            summary_rows.append({
                                "category_url": cat_url,
                                "category_title": item["category_title"],
                                "mot_cle_principal": item["category_main_term"],
                                "nb_produits": n_total,
                                "nb_suspects": n_sus,
                                "nb_a_exclure": n_excl,
                                "date_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            update_exclusions_from_df(sh, cat_url, item["df"])
                        
                        write_df_to_sheet(sh, "SUMMARY", pd.DataFrame(summary_rows))
                        
                        if all_suspects_rows:
                            write_df_to_sheet(sh, "SUSPECTS", pd.concat(all_suspects_rows, ignore_index=True))
                        else:
                            write_df_to_sheet(sh, "SUSPECTS", pd.DataFrame({"info": ["Aucun suspect"]}))
                        
                        if all_products_rows:
                            df_all_products = pd.concat(all_products_rows, ignore_index=True)
                            preferred = [
                                "category_url", "category_title", "mot_cle_principal",
                                "nom", "marque", "prix", "prix_num",
                                "anomalie_score", "niveau_anomalie", "suspect",
                                "score_fiche_technique", "ano_fiche_technique",
                                "score_description", "ano_description",
                                "raison_suspecte", "reco_action",
                                "decision_client", "exclure_prochaine_analyse",
                                "url", "image_url",
                            ]
                            cols = [c for c in preferred if c in df_all_products.columns] + \
                                   [c for c in df_all_products.columns if c not in preferred]
                            write_df_to_sheet(sh, "ALL_PRODUCTS", df_all_products[cols])
                        
                        st.success("✅ Export Google Sheets terminé!")
                        
                except Exception as e:
                    st.error(f"❌ Erreur export: {e}")

# ===================================================
# RESULTS DISPLAY
# ===================================================

if st.session_state.results_by_category:
    st.markdown("---")
    st.markdown("## 📊 Résultats")
    
    # Stats globales
    all_dfs = [item["df"] for item in st.session_state.results_by_category if not item["df"].empty]
    if all_dfs:
        df_combined = pd.concat(all_dfs, ignore_index=True)
        total_products = len(df_combined)
        total_suspects = df_combined["suspect"].sum() if "suspect" in df_combined.columns else 0
        avg_score = df_combined["anomalie_score"].mean() if "anomalie_score" in df_combined.columns else 0
        
        render_stats_cards(
            total_products=total_products,
            total_suspects=int(total_suspects),
            categories_count=len(st.session_state.results_by_category),
            avg_score=avg_score
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Download global
        csv_global = df_combined.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger CSV complet",
            csv_global,
            file_name="produits_sfactory_analyse.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Par catégorie
    for item in st.session_state.results_by_category:
        cat_url = item["category_url"]
        title = item["category_title"] or cat_url
        df_final = item["df"]
        
        with st.expander(f"📁 {title}", expanded=False):
            st.markdown(f"**URL:** `{cat_url}`")
            st.markdown(f"**Mot-clé principal:** `{item['category_main_term']}`")
            
            if df_final.empty:
                st.info("Aucune donnée disponible.")
                continue
            
            # Mini stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Produits", len(df_final))
            with col2:
                suspects_count = df_final["suspect"].sum() if "suspect" in df_final.columns else 0
                st.metric("Suspects", int(suspects_count))
            with col3:
                avg = df_final["anomalie_score"].mean() if "anomalie_score" in df_final.columns else 0
                st.metric("Score moyen", f"{avg:.1f}")
            with col4:
                if "prix_num" in df_final.columns:
                    avg_price = df_final["prix_num"].mean()
                    st.metric("Prix moyen", f"{avg_price:.2f}€" if not pd.isna(avg_price) else "N/A")
            
            st.markdown("---")
            
            # Tableau principal
            display_cols = [
                "nom", "sku", "prix", "anomalie_score", "niveau_anomalie", 
                "suspect", "raison_suspecte", "reco_action", "url"
            ]
            display_cols = [c for c in display_cols if c in df_final.columns]
            
            st.dataframe(
                df_final[display_cols].style.apply(
                    lambda row: ['background-color: rgba(239, 68, 68, 0.2)' if row.get('suspect', False) else '' for _ in row],
                    axis=1
                ),
                use_container_width=True,
                height=400
            )
            
            # Suspects highlight
            if "suspect" in df_final.columns:
                suspects_df = df_final[df_final["suspect"] == True].sort_values("anomalie_score", ascending=False)
                if not suspects_df.empty:
                    st.markdown("#### ⚠️ Produits suspects")
                    st.dataframe(suspects_df[display_cols], use_container_width=True)
            
            # Download
            csv = df_final.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"📥 Télécharger CSV ({title[:30]}...)",
                csv,
                file_name=f"produits_{get_category_slug(cat_url) or 'categorie'}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-muted); font-size: 0.85rem; padding: 1rem;">
    S-Factory Product Analyzer • Powered by Python & Streamlit
</div>
""", unsafe_allow_html=True)