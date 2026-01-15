import io
import os
import re
import json
import glob
import time  # ✅ AJOUT (heartbeat)
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse
from collections import Counter

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image, ImageFilter, ImageOps
import imagehash  # garde pour le fallback pHash


# minoré l'impact de la longueur du texte, Le prix peut etre un indicateur pour valider
# que le produit est bien en erreur
# Pas d'impact sur le nombre de caractere de la description
# Augmenter le pourcentage de detecteur d'anomalie , trop de remontés en ano
# pour le titre , prendre seulement les mots les plus courants des titres produits de la cat et non du descriptif de la catégorie
# dans la description du produits, les mots clés les plus courants doivent se retrouver dans la description du produits
# Augmenter le poids sur la fiche technique  ( les erreurs sont souvents ici )
# ajouter une colonne anomalie juste fiche technique , ajouter une colonne ano description , si doute comparer les deux


# ==== CLIP : import sécurisé (pour ne pas planter si torch n'est pas installé) ====
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
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

STOPWORDS_FR = list({
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

st.set_page_config(page_title="Scraper S-Factory + Similarité", layout="wide")

if "results_by_category" not in st.session_state:
    st.session_state.results_by_category = []  # list[dict]


# ==== état global CLIP ====
CLIP_MODEL = None
CLIP_PREPROCESS = None
CLIP_DEVICE = "cpu"


def init_clip():
    global CLIP_MODEL, CLIP_PREPROCESS, CLIP_DEVICE
    if not HAS_CLIP or CLIP_MODEL is not None:
        return
    CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    model.to(CLIP_DEVICE)
    model.eval()
    CLIP_MODEL = model
    CLIP_PREPROCESS = preprocess


# ===================================================
# SHEETS HELPERS
# ===================================================
def sheets_client_from_streamlit_secrets():
    """
    Streamlit Cloud / local secrets.toml:
    st.secrets["gcp_service_account"] doit contenir le dict du service account.
    """
    sa_info = st.secrets.get("gcp_service_account", None)
    if not sa_info:
        raise RuntimeError("Secret manquant: st.secrets['gcp_service_account']")
    creds = Credentials.from_service_account_info(sa_info, scopes=SHEETS_SCOPES)
    return gspread.authorize(creds)


def sheets_client_from_service_account_file(sa_json_path: str):
    """
    Mode local uniquement: JSON sur disque.
    """
    creds = Credentials.from_service_account_file(sa_json_path, scopes=SHEETS_SCOPES)
    return gspread.authorize(creds)


SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def extract_sheet_id(sheet_url_or_id: str) -> str:
    s = (sheet_url_or_id or "").strip()
    if not s:
        return ""
    if "docs.google.com" in s and "/spreadsheets/d/" in s:
        return s.split("/spreadsheets/d/")[1].split("/")[0]
    return s


def guess_service_account_json(default_dir: str = ".") -> str:
    candidates = glob.glob(os.path.join(default_dir, "*.json"))
    for c in candidates:
        try:
            with open(c, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("type") == "service_account" and "client_email" in data:
                return c
        except Exception:
            continue
    return ""


def sheets_client_from_service_account(sa_json_path: str):
    creds = Credentials.from_service_account_file(sa_json_path, scopes=SHEETS_SCOPES)
    return gspread.authorize(creds)


def sanitize_sheet_title(title: str, max_len: int = 90) -> str:
    t = re.sub(r"[\[\]\:\*\?\/\\]", " ", title).strip()
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        t = "Categorie"
    return t[:max_len]


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
        if not values:
            return pd.DataFrame()
        header = values[0]
        rows = values[1:]
        if not header:
            return pd.DataFrame()
        return pd.DataFrame(rows, columns=header)
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
    if df_excl.empty:
        return set()
    if "category_url" not in df_excl.columns or "product_url" not in df_excl.columns:
        return set()
    df_excl = df_excl[df_excl["category_url"].astype(str) == str(category_url)]
    urls = df_excl["product_url"].astype(str).tolist()
    return set(u for u in urls if u and u != "nan")


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
    if excl.empty:
        return

    rows = excl.astype(str).values.tolist()
    ws.append_rows(rows, value_input_option="USER_ENTERED")


def load_decisions_map_from_all_products(sh) -> dict:
    """
    Relit l’onglet ALL_PRODUCTS (si présent) et construit un mapping:
    product_url -> decision_client
    Ça permet de ne pas écraser les choix du client au run suivant.
    """
    df = read_sheet_as_df(sh, "ALL_PRODUCTS")
    if df.empty:
        return {}
    if "url" not in df.columns or "decision_client" not in df.columns:
        return {}

    m = {}
    for u, d in zip(df["url"].astype(str), df["decision_client"].astype(str)):
        u = u.strip()
        if u:
            m[u] = d.strip()
    return m


def delete_all_tabs_except(sh, keep_titles: set):
    """
    Supprime tous les onglets sauf ceux listés.
    Attention: le tableur doit garder au moins 1 onglet.
    On force donc la création des onglets 'keep' avant suppression.
    """
    for t in keep_titles:
        ensure_worksheet(sh, t)

    worksheets = sh.worksheets()
    for ws in worksheets:
        if ws.title in keep_titles:
            continue
        try:
            sh.del_worksheet(ws)
        except Exception:
            # si un onglet refuse de mourir, on ne bloque pas tout le run
            pass


# ===================================================
# UTILS HTTP / HTML
# ===================================================

def absolutize_url(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return BASE_DOMAIN.rstrip("/") + "/" + href.lstrip("/")


def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


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
    if h1:
        return h1.get_text(" ", strip=True)
    return ""


def extract_category_intro_text(soup: BeautifulSoup, max_chars: int = 4000) -> str:
    selectors = [
        ".category-description",
        "#category-description",
        ".category-description.std",
        ".std",
        ".category-view .category-description",
    ]
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            txt = node.get_text(" ", strip=True)
            txt = re.sub(r"\s+", " ", txt).strip()
            if len(txt) > 50:
                return txt[:max_chars]

    main = soup.select_one("main") or soup.select_one("#main") or soup.select_one(".main") or soup.body
    if not main:
        return ""

    for kill_sel in [".category-products", ".products-grid", ".products", "ol.products", "ul.products-grid"]:
        for k in main.select(kill_sel):
            k.decompose()

    paragraphs = main.find_all(["p", "li"], limit=30)
    parts = []
    for p in paragraphs:
        t = p.get_text(" ", strip=True)
        t = re.sub(r"\s+", " ", t).strip()
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
# ✅ PATCH MINIMAL ICI (get_product_urls)
# ===================================================

def get_product_urls(category_url: str, max_pages: int = 80) -> list:
    """
    PATCH:
    - Ne filtre plus les URLs produit avec (slug in href) => évite de rater des produits listés.
    - Utilise des sélecteurs produits (Magento) en priorité.
    - Fallback sur tous les liens .html si aucun sélecteur ne retourne de résultat.
    """
    urls = set()
    visited_pages = set()
    current = category_url

    for _ in range(max_pages):
        if current in visited_pages:
            break
        visited_pages.add(current)

        soup = get_soup(current)

        # 1) Sélecteurs produits (priorité)
        selectors = [
            "a.product-item-link[href]",
            "h2.product-name a[href]",
            ".product-name a[href]",
            "a.product-image[href]",
        ]
        page_urls = set()
        for sel in selectors:
            for a in soup.select(sel):
                href = a.get("href", "")
                if href and href.endswith(".html"):
                    page_urls.add(absolutize_url(href))

        # 2) Fallback: tout .html (sans filtrer par slug)
        if not page_urls:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href and href.endswith(".html"):
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

def extract_caracteristiques(soup: BeautifulSoup) -> list:
    h_carac = None
    for tag in soup.find_all(["h2", "h3"]):
        text = tag.get_text(strip=True)
        if "Caractéristiques" in text:
            h_carac = tag
            break
    if not h_carac:
        return []
    ul = h_carac.find_next("ul")
    if not ul:
        return []
    return [li.get_text(strip=True) for li in ul.find_all("li")]


def extract_price(soup: BeautifulSoup) -> str:
    special = soup.find("p", class_="special-price")
    if special:
        span = special.find("span", class_="price")
        if span:
            return span.get_text(strip=True)
    span = soup.find("span", class_="price")
    if span:
        return span.get_text(strip=True)
    return ""


def extract_image_url(soup: BeautifulSoup) -> str:
    img = soup.find("img", attrs={"data-src": True})
    if img:
        url = img.get("data-src")
        if url:
            return absolutize_url(url)

    img = soup.find("img", attrs={"data-original": True})
    if img:
        url = img.get("data-original")
        if url:
            return absolutize_url(url)

    product_imgs = soup.select("img[src*='catalog'], img[src*='product'], img[src*='media']")
    for img in product_imgs:
        src = img.get("src")
        if src and len(src) > 10:
            return absolutize_url(src)

    img = soup.find("img")
    if img and img.get("src"):
        return absolutize_url(img.get("src"))

    return ""


def extract_tech_specs(soup: BeautifulSoup) -> dict:
    h_info = None
    for tag in soup.find_all(["h2", "h3"]):
        text = tag.get_text(strip=True).upper()
        if "INFORMATIONS" in text and "COMPL" in text:
            h_info = tag
            break
    if not h_info:
        return {}
    table = h_info.find_next("table")
    if not table:
        return {}
    specs = {}
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) >= 2:
            key = cells[0].get_text(strip=True)
            val = cells[1].get_text(strip=True)
            if key:
                specs[key] = val
    return specs


# ===================================================
# ✅ AJOUTS DEMANDÉS (SKU + images description + title qty/size)
# ===================================================

SKU_REGEX = re.compile(r"\b([A-Z]{3,5}-\d{4})\b")


def extract_sku(soup: BeautifulSoup) -> str:
    """
    SKU souvent sous le titre, format ex: FERO-1034 / AVAP-0087.
    On scanne autour du H1 + bloc produit pour être robuste.
    """
    h1 = soup.find("h1")
    if h1:
        zone = h1.parent.get_text(" ", strip=True) if h1.parent else h1.get_text(" ", strip=True)
        m = SKU_REGEX.search(zone)
        if m:
            return m.group(1)

    txt = soup.get_text(" ", strip=True)
    m = SKU_REGEX.search(txt)
    return m.group(1) if m else ""


def extract_description_block(soup: BeautifulSoup):
    """
    Retourne:
    - description_txt : texte propre
    - description_html : html brut du bloc
    - desc_img_urls : liste d'URLs d'images trouvées dans le bloc description
    """
    h_desc = None
    for tag in soup.find_all(["h2", "h3"]):
        if "Description" in tag.get_text(strip=True):
            h_desc = tag
            break
    if not h_desc:
        return "", "", []

    nodes = []
    for sib in h_desc.next_siblings:
        name = getattr(sib, "name", None)
        if name in ("h2", "h3"):
            break
        if getattr(sib, "name", None) is not None:
            nodes.append(sib)

    if not nodes:
        return "", "", []

    description_html = "".join(str(n) for n in nodes)

    parts = []
    for n in nodes:
        txt = n.get_text(" ", strip=True) if hasattr(n, "get_text") else ""
        if txt:
            parts.append(txt)
    description_txt = " ".join(parts)

    desc_img_urls = []
    for n in nodes:
        for img in n.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-original")
            if src:
                desc_img_urls.append(absolutize_url(src))

    seen = set()
    desc_img_urls = [u for u in desc_img_urls if not (u in seen or seen.add(u))]

    return description_txt, description_html, desc_img_urls


def check_image_url_ok(url: str) -> (bool, str):
    """
    Check simple: url valide + request OK + content-type image.
    """
    if not url or not isinstance(url, str):
        return False, "url vide"

    try:
        r = requests.get(url, headers=HEADERS, timeout=10, stream=True)
        if r.status_code >= 400:
            return False, f"http {r.status_code}"
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "image" not in ctype:
            return False, f"content-type={ctype or 'inconnu'}"
        return True, "ok"
    except Exception as e:
        return False, f"erreur: {e}"


def title_qty_size_signals(title: str) -> dict:
    """
    Repérage dans le titre:
    - quantité: X48, x 48, lot de 3, pack 10
    - taille: 23CM, 23 CM, 0.5L, 500ML, 1.2M, 10MM, 2G, etc.
    """
    t = (title or "").strip()

    qty = None
    m = re.search(r"(?i)\b[x×]\s?(\d{1,4})\b", t)
    if m:
        qty = int(m.group(1))
    else:
        m = re.search(r"(?i)\b(lot|pack)\s*(de)?\s*(\d{1,4})\b", t)
        if m:
            qty = int(m.group(3))

    size_raw = ""
    size_val = None
    size_unit = ""

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

    desc_img_ok = []
    desc_img_reason = []
    desc_img_sim_to_main = []

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
                    sim = ""
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
        col_name = "fiche_" + (
            k.strip()
             .lower()
             .replace(" ", "_")
             .replace("'", "_")
             .replace("é", "e")
             .replace("è", "e")
             .replace("ê", "e")
             .replace("à", "a")
        )
        data[col_name] = v

    return data


# ===================================================
# ENRICHISSEMENT STRUCTURÉ
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

    term = (category_main_term or "").strip().lower()
    if not term:
        term = "produit"

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

    # (on garde ces colonnes, mais elles ne pilotent plus l'anomalie fiche technique)
    fiche_cols = [c for c in df.columns if c.startswith("fiche_")]
    df["nb_fiche_champs"] = len(fiche_cols)
    if fiche_cols:
        df["nb_fiche_non_vides"] = df[fiche_cols].apply(
            lambda row: sum(bool(str(v).strip()) for v in row),
            axis=1
        )
        df["taux_fiche_completude"] = df["nb_fiche_non_vides"] / df["nb_fiche_champs"]
    else:
        df["nb_fiche_non_vides"] = 0
        df["taux_fiche_completude"] = np.nan

    return df


# ===================================================
# ✅ NOUVEAU: COHÉRENCE FICHE TECHNIQUE (comparaison valeurs)
# ===================================================

def _norm_txt(v) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’", "'")
    return s


def _parse_dimensions_to_mm(val: str):
    s = _norm_txt(val)
    if not s:
        return None
    s = s.replace("×", "x").replace("*", "x")
    unit = "mm"
    if "cm" in s:
        unit = "cm"
    nums = re.findall(r"(\d+(?:[.,]\d+)?)", s)
    if len(nums) < 2:
        return None
    a = float(nums[0].replace(",", "."))
    b = float(nums[1].replace(",", "."))
    if unit == "cm":
        a *= 10
        b *= 10
    return (a, b)


def _tech_field_consensus(series: pd.Series, min_ratio: float = 0.6):
    vals = [_norm_txt(x) for x in series.tolist()]
    vals = [v for v in vals if v and v != "nan"]
    if not vals:
        return ("", 0.0)
    c = Counter(vals)
    top_val, top_cnt = c.most_common(1)[0]
    ratio = top_cnt / max(1, len(vals))
    if ratio >= min_ratio:
        return (top_val, ratio)
    return ("", ratio)


def compute_fiche_technique_incoherence(df: pd.DataFrame, consensus_ratio: float = 0.6) -> pd.DataFrame:
    if df.empty:
        df["score_fiche_technique"] = 0
        df["ano_fiche_technique"] = False
        df["fiche_tech_issues"] = ""
        df["nb_fiche_champs_consideres"] = 0
        return df

    fiche_cols = [c for c in df.columns if c.startswith("fiche_")]
    if not fiche_cols:
        df["score_fiche_technique"] = 0
        df["ano_fiche_technique"] = False
        df["fiche_tech_issues"] = ""
        df["nb_fiche_champs_consideres"] = 0
        return df

    stable_cols = []
    consensus = {}
    for col in fiche_cols:
        # ✅ DEMANDE CLIENT: ignorer le champ "dimensions" dans l'analyse
        if "dimension" in col:
            continue

        top_val, ratio = _tech_field_consensus(df[col], min_ratio=consensus_ratio)
        if top_val:
            stable_cols.append(col)
            consensus[col] = (top_val, ratio)

    issues_list = []
    scores = []

    for _, row in df.iterrows():
        issues = []
        penalty = 0

        for col in stable_cols:
            v = _norm_txt(row.get(col, ""))
            if not v or v == "nan":
                continue
            top_val, _ = consensus[col]
            if v != top_val:
                penalty += 12
                issues.append(f"{col.replace('fiche_', '')}: '{v}' ≠ '{top_val}'")

        score = int(min(100, penalty))
        scores.append(score)
        issues_list.append(" ; ".join(issues))

    df["score_fiche_technique"] = scores
    df["ano_fiche_technique"] = df["score_fiche_technique"] >= 25
    df["fiche_tech_issues"] = issues_list
    df["nb_fiche_champs_consideres"] = len(stable_cols)

    # -------------------------
    # ✅ PATCH DEMANDÉ: rareté UNIQUEMENT sur la colonne "type" si présente
    # -------------------------
    # on reset un cache éventuel (au cas où plusieurs catégories)
    if hasattr(compute_fiche_technique_incoherence, "_cached_type_col"):
        delattr(compute_fiche_technique_incoherence, "_cached_type_col")

    # on cherche une colonne "type"
    type_candidates = [c for c in fiche_cols if "type" in c.lower()]
    type_col = ""
    if type_candidates:
        # priorité à fiche_type si elle existe
        for c in fiche_cols:
            if c.lower() == "fiche_type":
                type_col = c
                break
        if not type_col:
            type_col = type_candidates[0]

    if type_col:
        vals = df[type_col].fillna("").astype(str).apply(_norm_txt)
        vals = vals[vals != ""]
        vals = vals[vals != "nan"]
        if len(vals) > 0:
            c = Counter(vals.tolist())
            total = sum(c.values())
            rare_vals = {k for k, v in c.items() if (v / max(1, total)) <= 0.10 and v >= 1}

            # on ajoute la rareté dans les issues + score (sans toucher aux autres colonnes)
            new_scores = []
            new_issues = []
            for s, iss, v in zip(df["score_fiche_technique"].tolist(), df["fiche_tech_issues"].tolist(), df[type_col].fillna("").astype(str).tolist()):
                v_norm = _norm_txt(v)
                add_pen = 0
                add_txt = ""
                if v_norm and v_norm in rare_vals:
                    add_pen = 25
                    add_txt = f"{type_col.replace('fiche_', '')}: valeur rare '{v_norm}'"
                ns = int(min(100, (s or 0) + add_pen))
                if add_txt:
                    if iss:
                        ni = iss + " ; " + add_txt
                    else:
                        ni = add_txt
                else:
                    ni = iss
                new_scores.append(ns)
                new_issues.append(ni)

            df["score_fiche_technique"] = new_scores
            df["fiche_tech_issues"] = new_issues
            df["ano_fiche_technique"] = df["score_fiche_technique"] >= 25

    return df


# ===================================================
# MOTS-CLES (titres produits) + couverture description
# ===================================================

def tokenize_fr(text: str) -> list:
    text = (text or "").lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    toks = [t for t in text.split(" ") if t and t not in STOPWORDS_FR and len(t) > 3]
    return toks


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
    hit = sum(1 for k in keywords if k in toks)
    return hit / max(1, len(keywords))


# ===================================================
# SIMILARITÉ TEXTE (produit-catégorie) ✅ sans texte global
# ===================================================

def compute_similarity(df: pd.DataFrame, category_intro_text: str) -> pd.DataFrame:
    """
    Version "sans texte global":
    - On NE calcule PLUS similarite_moyenne (produit-produit).
    - On garde similarite_categorie: (titre + description) vs texte catégorie.
    """
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

    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS_FR,
        ngram_range=(1, 2),
        min_df=1
    )
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
            continue
        top_idx = row.argsort()[-7:][::-1]
        terms = [t for t in feature_names[top_idx] if len(t) > 2]
        main_terms.append(", ".join(terms[:5]))

    df["mots_cles_principaux"] = main_terms
    return df


# ===================================================
# IMAGE : CLIP/pHash + forme/couleur/global (ZOOM OBJET)
# ===================================================

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return np.nan
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def download_pil_image(url: str):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
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

    y0 = max(0, y0 - dy)
    y1 = min(h - 1, y1 + dy)
    x0 = max(0, x0 - dx)
    x1 = min(w - 1, x1 + dx)

    return img.crop((x0, y0, x1 + 1, y1 + 1))


def prepare_image_for_analysis(img: Image.Image) -> Image.Image:
    img2 = crop_to_object(img, white_thresh=245, margin=0.10)
    img2 = img2.resize((256, 256))
    return img2


def color_signature(img: Image.Image, bins_h=24, bins_s=8, bins_v=8) -> np.ndarray:
    img = prepare_image_for_analysis(img)

    hsv = img.convert("HSV")
    arr = np.asarray(hsv, dtype=np.uint8)
    h = arr[:, :, 0].astype(np.int32)
    s = arr[:, :, 1].astype(np.int32)
    v = arr[:, :, 2].astype(np.int32)

    h_bin = (h * bins_h) // 256
    s_bin = (s * bins_s) // 256
    v_bin = (v * bins_v) // 256

    hist = np.zeros((bins_h, bins_s, bins_v), dtype=np.float32)
    np.add.at(hist, (h_bin, s_bin, v_bin), 1.0)
    return _l2_normalize(hist.ravel())


def shape_signature(img: Image.Image, size=128) -> np.ndarray:
    img = prepare_image_for_analysis(img)

    gray = ImageOps.grayscale(img).resize((size, size))
    edges = gray.filter(ImageFilter.FIND_EDGES)

    arr = np.asarray(edges, dtype=np.uint8).astype(np.float32)
    arr = np.clip(arr * 1.5, 0, 255)

    hist, _ = np.histogram(arr.ravel(), bins=32, range=(0, 255))
    return _l2_normalize(hist.astype(np.float32))


def phash_similarity(h1, h2) -> float:
    if h1 is None or h2 is None:
        return np.nan
    dist = h1 - h2
    return float(1 - (dist / 64))


def get_clip_embedding(url: str):
    if not HAS_CLIP:
        return None
    init_clip()
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img_t = CLIP_PREPROCESS(img).unsqueeze(0).to(CLIP_DEVICE)
        with torch.no_grad():
            emb = CLIP_MODEL.encode_image(img_t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]
    except Exception:
        return None


def compute_image_similarity_phash(df: pd.DataFrame) -> pd.DataFrame:
    hashes = []
    shape_vecs = []
    color_vecs = []

    for url in df["image_url"]:
        img = download_pil_image(url) if isinstance(url, str) and url.strip() else None

        if img is None:
            hashes.append(None)
            shape_vecs.append(None)
            color_vecs.append(None)
            continue

        try:
            hashes.append(imagehash.phash(img))
        except Exception:
            hashes.append(None)

        try:
            shape_vecs.append(shape_signature(img))
        except Exception:
            shape_vecs.append(None)

        try:
            color_vecs.append(color_signature(img))
        except Exception:
            color_vecs.append(None)

    df["image_hash"] = [str(h) if h is not None else None for h in hashes]
    df["image_embedding"] = None
    df["image_shape_vec"] = shape_vecs
    df["image_color_vec"] = color_vecs

    sim_img_mean = []
    sim_shape_mean = []
    sim_color_mean = []
    sim_global_mean = []

    n = len(df)
    for i in range(n):
        sims_ph, sims_shape, sims_color, sims_global = [], [], [], []
        for j in range(n):
            if i == j:
                continue

            s_ph = phash_similarity(hashes[i], hashes[j])
            s_sh = _cosine(shape_vecs[i], shape_vecs[j])
            s_col = _cosine(color_vecs[i], color_vecs[j])

            if not np.isnan(s_ph):
                sims_ph.append(s_ph)
            if not np.isnan(s_sh):
                sims_shape.append(s_sh)
            if not np.isnan(s_col):
                sims_color.append(s_col)

            parts, weights = [], []
            if not np.isnan(s_ph):
                parts.append(s_ph)
                weights.append(0.45)
            if not np.isnan(s_sh):
                parts.append(s_sh)
                weights.append(0.30)
            if not np.isnan(s_col):
                parts.append(s_col)
                weights.append(0.25)

            if parts and sum(weights) > 0:
                sims_global.append(float(np.average(parts, weights=weights)))

        sim_img_mean.append(float(np.mean(sims_ph)) if sims_ph else np.nan)
        sim_shape_mean.append(float(np.mean(sims_shape)) if sims_shape else np.nan)
        sim_color_mean.append(float(np.mean(sims_color)) if sims_color else np.nan)
        sim_global_mean.append(float(np.mean(sims_global)) if sims_global else np.nan)

    df["similarite_image_moyenne"] = sim_img_mean
    df["similarite_forme_moyenne"] = sim_shape_mean
    df["similarite_couleur_moyenne"] = sim_color_mean
    df["similarite_image_globale_moyenne"] = sim_global_mean

    return df


def compute_image_similarity_clip(df: pd.DataFrame) -> pd.DataFrame:
    embeddings = []
    shape_vecs = []
    color_vecs = []

    for url in df["image_url"]:
        img = download_pil_image(url) if isinstance(url, str) and url.strip() else None

        if img is None:
            embeddings.append(None)
            shape_vecs.append(None)
            color_vecs.append(None)
            continue

        embeddings.append(get_clip_embedding(url))

        try:
            shape_vecs.append(shape_signature(img))
        except Exception:
            shape_vecs.append(None)

        try:
            color_vecs.append(color_signature(img))
        except Exception:
            color_vecs.append(None)

    df["image_embedding"] = embeddings
    df["image_hash"] = None
    df["image_shape_vec"] = shape_vecs
    df["image_color_vec"] = color_vecs

    n = len(df)
    sim_clip_mean, sim_shape_mean, sim_color_mean, sim_global_mean = [], [], [], []

    for i in range(n):
        sims_clip, sims_shape, sims_color, sims_global = [], [], [], []
        for j in range(n):
            if i == j:
                continue

            if embeddings[i] is not None and embeddings[j] is not None:
                s_clip = float(np.dot(embeddings[i], embeddings[j]))
                sims_clip.append(s_clip)
            else:
                s_clip = np.nan

            s_sh = _cosine(shape_vecs[i], shape_vecs[j])
            if not np.isnan(s_sh):
                sims_shape.append(s_sh)

            s_col = _cosine(color_vecs[i], color_vecs[j])
            if not np.isnan(s_col):
                sims_color.append(s_col)

            parts, weights = [], []
            if not np.isnan(s_clip):
                parts.append(s_clip)
                weights.append(0.55)
            if not np.isnan(s_sh):
                parts.append(s_sh)
                weights.append(0.25)
            if not np.isnan(s_col):
                parts.append(s_col)
                weights.append(0.20)

            if parts and sum(weights) > 0:
                sims_global.append(float(np.average(parts, weights=weights)))

        sim_clip_mean.append(float(np.mean(sims_clip)) if sims_clip else np.nan)
        sim_shape_mean.append(float(np.mean(sims_shape)) if sims_shape else np.nan)
        sim_color_mean.append(float(np.mean(sims_color)) if sims_color else np.nan)
        sim_global_mean.append(float(np.mean(sims_global)) if sims_global else np.nan)

    df["similarite_image_moyenne"] = sim_clip_mean
    df["similarite_forme_moyenne"] = sim_shape_mean
    df["similarite_couleur_moyenne"] = sim_color_mean
    df["similarite_image_globale_moyenne"] = sim_global_mean

    return df


def compute_image_similarity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["image_hash"] = None
        df["image_embedding"] = None
        df["similarite_image_moyenne"] = np.nan
        df["similarite_forme_moyenne"] = np.nan
        df["similarite_couleur_moyenne"] = np.nan
        df["similarite_image_globale_moyenne"] = np.nan
        return df

    if HAS_CLIP:
        return compute_image_similarity_clip(df)
    return compute_image_similarity_phash(df)


# ===================================================
# FLAGS / SCORE
# ===================================================

def add_outlier_flags_and_reasons(df: pd.DataFrame, category_main_term: str) -> pd.DataFrame:
    if df.empty:
        df["anomalie_score"] = 0
        df["niveau_anomalie"] = "OK"
        df["suspect"] = False
        df["raison_suspecte"] = ""
        df["reco_action"] = ""
        df["score_fiche_technique"] = 0
        df["ano_fiche_technique"] = False
        df["score_description"] = 0
        df["ano_description"] = False
        return df

    # ✅ DEMANDE CLIENT: suppression du "texte global" => on ne calcule plus similarite_moyenne
    threshold_img = float(np.nanquantile(df["similarite_image_moyenne"], 0.05)) if "similarite_image_moyenne" in df.columns and df["similarite_image_moyenne"].notna().any() else None
    threshold_cat = float(np.nanquantile(df["similarite_categorie"], 0.10)) if "similarite_categorie" in df.columns and df["similarite_categorie"].notna().any() else None

    if "prix_num" in df.columns and df["prix_num"].notna().any():
        q1 = float(df["prix_num"].quantile(0.25))
        q3 = float(df["prix_num"].quantile(0.75))
    else:
        q1 = q3 = None

    majority_portable = (df["is_portable"].mean() > 0.5 if "is_portable" in df.columns else False)

    scores, niveaux, suspects_flags, reasons, actions = [], [], [], [], []
    score_ft_list, ano_ft_list = [], []
    score_desc_list, ano_desc_list = [], []

    term = (category_main_term or "produit").strip()

    for _, row in df.iterrows():
        row_reasons = []
        row_actions = []
        score = 0
        score_ft = 0
        score_desc = 0

        if threshold_cat is not None:
            simc = row.get("similarite_categorie", np.nan)
            if isinstance(simc, (int, float)) and not np.isnan(simc) and simc < threshold_cat:
                score += 20
                row_reasons.append("texte produit peu aligné avec l'introduction de la catégorie")
                row_actions.append("Vérifier si le produit est bien rangé dans cette catégorie (ou si le contenu est hors-sujet)")

        if threshold_img is not None:
            sim_img = row.get("similarite_image_moyenne", np.nan)
            if isinstance(sim_img, (int, float)) and not np.isnan(sim_img) and sim_img < threshold_img:
                score += 20
                row_reasons.append("image très différente des autres produits de la catégorie")
                row_actions.append("Vérifier si l’image correspond bien au produit et à la catégorie")

        # ✅ DEMANDE CLIENT: vérif image(s) incluse(s) dans la description (balise cassée / hors sujet)
        desc_ok = str(row.get("desc_img_ok", "")).lower()
        desc_sims = str(row.get("desc_img_sim_to_main", ""))

        if desc_ok and desc_ok != "nan":
            oks = [x.strip() for x in desc_ok.split("|") if x.strip()]
            bad = any(x in ("false", "0") for x in oks)
            if bad:
                score += 15
                row_reasons.append("image(s) dans la description cassée(s) / non chargée(s)")
                row_actions.append("Corriger la/les balises <img> dans la description (src, media, cache, etc.)")

        try:
            sims = []
            for s in desc_sims.split("|"):
                s = s.strip()
                if s and s not in ("nan", ""):
                    sims.append(float(s))
            if sims and min(sims) < 0.35:
                score += 10
                row_reasons.append("image(s) intégrée(s) à la description potentiellement hors-sujet")
                row_actions.append("Vérifier l’image intégrée dans la description (cohérence produit)")
        except Exception:
            pass

        is_capsule = row.get("has_capsule_word", False)
        is_accessoire = row.get("has_accessoire_word", False)
        has_cat_word = row.get("has_category_word", True)

        if is_capsule:
            score += 25
            row_reasons.append("semble être une capsule plutôt qu'un produit principal")
            row_actions.append("Déplacer en catégorie 'capsules' ou 'accessoires' si disponible")

        if is_accessoire:
            score += 25
            row_reasons.append("contient des termes d'accessoire (adaptateur, embout, pièce détachée, etc.)")
            row_actions.append("Déplacer en catégorie 'accessoires' ou créer une sous-catégorie dédiée")

        if not has_cat_word:
            score += 15
            row_reasons.append(f"ne contient pas le mot-clé principal de catégorie ('{term}') dans le titre / description")
            row_actions.append("Enrichir le titre/description avec le bon type de produit ou vérifier le classement catégorie")

        if majority_portable and row.get("is_salon", False):
            score += 15
            row_reasons.append("mentionne 'de salon' alors que la majorité des produits semble portable")
            row_actions.append("Vérifier si ce modèle ne devrait pas être dans une autre catégorie (produits de salon)")

        # ---- ✅ NOUVEAU: fiche technique incohérente (comparaison valeurs) ----
        ft_score_raw = row.get("score_fiche_technique", 0)
        if isinstance(ft_score_raw, (int, float)) and not np.isnan(ft_score_raw) and ft_score_raw > 0:
            if ft_score_raw >= 40:
                score_ft += 25
                row_reasons.append("fiche technique incohérente vs les autres produits de la catégorie")
                det = row.get("fiche_tech_issues", "")
                if det:
                    row_reasons.append(f"détails: {det}")
                row_actions.append("Vérifier les valeurs de la fiche technique (type, format, etc.)")
            elif ft_score_raw >= 25:
                score_ft += 12
                row_reasons.append("fiche technique légèrement incohérente vs catégorie")
                row_actions.append("Contrôler la fiche technique sur quelques champs")

        # ---- description: couverture des mots-clés titres catégorie ----
        cov = row.get("desc_kw_coverage", np.nan)
        if isinstance(cov, (int, float)) and not np.isnan(cov) and cov < 0.20:
            score_desc += 15
            row_reasons.append("description peu alignée avec les mots courants des titres de la catégorie")
            row_actions.append("Enrichir la description avec les termes attendus (issus des titres produits)")

        score += score_ft + score_desc

        # ---- prix comme indicateur de confirmation (pas déclencheur principal) ----
        prix = row.get("prix_num")
        if score >= 30 and q1 is not None and isinstance(prix, (int, float)) and not np.isnan(prix):
            if prix < q1 or prix > q3:
                score += 10
                row_reasons.append("prix atypique (renforce le doute)")
                row_actions.append("Vérifier cohérence prix vs catégorie")

        score = min(score, 100)

        if score < 30:
            niveau = "OK"
        elif score < 50:
            niveau = "faible"
        elif score < 70:
            niveau = "moyenne"
        else:
            niveau = "forte"

        suspect = score >= 65

        reasons_text = " ; ".join(dict.fromkeys(r for r in row_reasons if r)) or ("Aucune anomalie détectée en priorité" if niveau == "OK" else "")
        action_text = " | ".join(dict.fromkeys(a for a in row_actions if a)) or "OK – aucun changement prioritaire recommandé"

        scores.append(score)
        niveaux.append(niveau)
        suspects_flags.append(suspect)
        reasons.append(reasons_text)
        actions.append(action_text)

        score_ft_list.append(int(score_ft))
        ano_ft_list.append(bool(ft_score_raw >= 25))
        score_desc_list.append(int(score_desc))
        ano_desc_list.append(bool(score_desc >= 15))

    df["anomalie_score"] = scores
    df["niveau_anomalie"] = niveaux
    df["suspect"] = suspects_flags
    df["raison_suspecte"] = reasons
    df["reco_action"] = actions

    df["score_fiche_technique"] = score_ft_list
    df["ano_fiche_technique"] = ano_ft_list
    df["score_description"] = score_desc_list
    df["ano_description"] = ano_desc_list

    return df


# ===================================================
# COLONNES CLIENT (choix + exclusion)
# ===================================================

def add_client_decision_columns(df: pd.DataFrame, decisions_map: Optional[dict] = None) -> pd.DataFrame:
    if df.empty:
        df["decision_client"] = ""
        df["exclure_prochaine_analyse"] = False
        return df

    if "decision_client" not in df.columns:
        df["decision_client"] = ""

    # réappliquer les décisions du client si elles existent déjà dans ALL_PRODUCTS
    if decisions_map:
        urls = df["url"].astype(str).tolist()
        df["decision_client"] = [
            (decisions_map.get(u.strip(), "") or "").strip() if u else ""
            for u in urls
        ]

    df["exclure_prochaine_analyse"] = df["decision_client"].astype(str).str.strip().str.lower().isin(
        ["a_exclure", "exclure", "exclude", "a exclure"]
    )
    return df


# ===================================================
# UI
# ===================================================

st.title("Scrapping et analyse de produits e-commerce")

if HAS_CLIP:
    st.caption("Analyse d'image : CLIP (embeddings visuels) + forme/couleur/global.")
else:
    st.caption("Analyse d'image : pHash (fallback) + forme/couleur/global. (Installe torch + open-clip-torch pour CLIP)")

with st.sidebar:
    st.subheader("Entrées")
    urls_text = st.text_area(
        "URLs catégories (1 par ligne)",
        value="https://www.planete-sfactory.com/vaporisateur/vaporisateur-portable/",
        height=140
    )
    analyse_image = st.checkbox("Activer l'analyse de similarité d'image", value=True)
    max_pages = st.number_input("Max pages de pagination (par catégorie)", min_value=1, max_value=200, value=80, step=1)

    st.divider()
    st.subheader("Export vers Sheets")
    export_sheets = st.checkbox("Exporter vers Sheets", value=True)

    sheet_url_default = "https://docs.google.com/spreadsheets/d/1lSqjJb-6HZQVfsDVrjeN2wULtFW7luK4d-HmBQOlx58/edit?gid=0#gid=0"
    sheet_url_or_id = st.text_input("URL ou ID du tableur", value=sheet_url_default)

    auth_mode = st.selectbox(
        "Mode d'auth Google",
        ["Streamlit secrets (recommandé)", "Fichier JSON local"],
        index=0
    )
    sa_json_path = ""
    if auth_mode == "Fichier JSON local":
        sa_json_path = st.text_input("Chemin JSON service account (local)", value="")
    else:
        st.caption("Utilise st.secrets['gcp_service_account'] (Streamlit Cloud ou .streamlit/secrets.toml en local).")

    wipe_tabs = st.checkbox(
        "Supprimer tous les onglets avant de réécrire (conserve EXCLUSIONS)",
        value=False
    )

    if HAS_SHEETS and sa_json_path:
        try:
            with open(sa_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sa_email = data.get("client_email", "")
            if sa_email:
                st.caption(f"Email service account : {sa_email}")
        except Exception:
            pass

run = st.button("Lancer le scraping & l'analyse (toutes catégories)")

# ✅ AJOUT: zones d'affichage "le code vit"
heartbeat = st.empty()
prod_status = st.empty()
prod_prog = st.empty()

if run:
    category_urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    if not category_urls:
        st.error("Mets au moins une URL de catégorie.")
    else:
        st.session_state.results_by_category = []

        spreadsheet_id = extract_sheet_id(sheet_url_or_id)
        sh = None
        sheets_error = None
        decisions_map = {}

        if export_sheets:
            if not HAS_SHEETS:
                sheets_error = "Packages manquants: installe gspread + google-auth"
            elif not spreadsheet_id:
                sheets_error = "ID du tableur introuvable (URL/ID invalide)."
            else:
                try:
                    if auth_mode == "Streamlit secrets (recommandé)":
                        client = sheets_client_from_streamlit_secrets()
                    else:
                        if not sa_json_path or not os.path.exists(sa_json_path):
                            raise FileNotFoundError("JSON service account introuvable (chemin invalide).")
                        client = sheets_client_from_service_account_file(sa_json_path)

                    sh = client.open_by_key(spreadsheet_id)

                    ensure_exclusions_sheet(sh)

                    if wipe_tabs:
                        delete_all_tabs_except(sh, keep_titles={"EXCLUSIONS"})

                    decisions_map = load_decisions_map_from_all_products(sh)

                except Exception as e:
                    sheets_error = str(e)

        prog = st.progress(0)
        status = st.empty()

        last_heartbeat = 0.0  # ✅ AJOUT

        for idx_cat, category_url in enumerate(category_urls, start=1):
            status.write(f"Catégorie {idx_cat}/{len(category_urls)} : {category_url}")

            # ✅ AJOUT heartbeat (catégorie)
            now = time.time()
            if now - last_heartbeat > 1.0:
                heartbeat.info(f"🔄 En cours… catégorie {idx_cat}/{len(category_urls)} – {datetime.now().strftime('%H:%M:%S')}")
                last_heartbeat = now

            try:
                with st.spinner("Lecture catégorie (titre + intro)..."):
                    soup_cat = get_soup(category_url)
                    category_title = extract_category_title(soup_cat)
                    category_intro = extract_category_intro_text(soup_cat)
                    category_main_term = build_category_main_term(category_title, category_url)

                with st.spinner("Récupération URLs produits (pagination incluse)..."):
                    product_urls = get_product_urls(category_url, max_pages=int(max_pages))

                # ✅ AJOUT: reset barre produit
                prod_status.write("")
                prod_prog.progress(0)

                # exclusions persistantes
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
                    results = []
                    total_prod = len(product_urls)

                    for i, prod_url in enumerate(product_urls):
                        # ✅ AJOUT heartbeat + progression produit
                        if i == 0 or (time.time() - last_heartbeat) > 1.0:
                            heartbeat.info(
                                f"🧱 Scraping produits… {i+1}/{total_prod} – catégorie {idx_cat}/{len(category_urls)} – {datetime.now().strftime('%H:%M:%S')}"
                            )
                            last_heartbeat = time.time()

                        prod_status.write(f"Produit {i+1}/{total_prod} : {prod_url}")
                        prod_prog.progress((i + 1) / max(1, total_prod))

                        try:
                            product = scrape_product(prod_url)
                        except Exception as e:
                            product = {
                                "url": prod_url,
                                "nom": "ERREUR",
                                "sku": "",
                                "marque": "",
                                "prix": "",
                                "caracteristiques": f"Erreur: {e}",
                                "description": "",
                                "description_html": "",
                                "desc_img_urls": "",
                                "desc_img_ok": "",
                                "desc_img_reason": "",
                                "desc_img_sim_to_main": "",
                                "image_url": "",
                                "title_qty": "",
                                "title_size_raw": "",
                                "title_size_value": "",
                                "title_size_unit": "",
                            }
                        results.append(product)

                    df = pd.DataFrame(results)

                    # mots-clés basés sur les TITRES produits de la catégorie
                    common_title_terms = compute_common_title_terms(df, top_k=12)
                    df["category_title_keywords"] = ", ".join(common_title_terms)
                    df["desc_kw_coverage"] = df["description"].fillna("").astype(str).apply(
                        lambda d: description_keyword_coverage(d, common_title_terms)
                    )

                    df = enrich_structured_features(df, category_main_term)

                    # ✅ compute_similarity sans texte global
                    df = compute_similarity(df, category_intro)

                    if analyse_image:
                        # ✅ AJOUT heartbeat (images)
                        heartbeat.info(f"🖼️ Analyse images… catégorie {idx_cat}/{len(category_urls)} – {datetime.now().strftime('%H:%M:%S')}")
                        with st.spinner("Analyse images..."):
                            df = compute_image_similarity(df)

                    # ✅ score incohérence fiche technique (comparaison valeurs), sans dimensions
                    df = compute_fiche_technique_incoherence(df, consensus_ratio=0.6)

                    df = add_outlier_flags_and_reasons(df, category_main_term)

                    # colonnes client + réapplique décisions existantes
                    df = add_client_decision_columns(df, decisions_map=decisions_map)

                st.session_state.results_by_category.append({
                    "category_url": category_url,
                    "category_title": category_title,
                    "category_main_term": category_main_term,
                    "category_intro": category_intro,
                    "df": df
                })

                prog.progress(idx_cat / len(category_urls))

            except Exception as e:
                st.error(f"Erreur sur la catégorie {category_url} : {e}")
                st.session_state.results_by_category.append({
                    "category_url": category_url,
                    "category_title": "",
                    "category_main_term": "produit",
                    "category_intro": "",
                    "df": pd.DataFrame()
                })
                prog.progress(idx_cat / len(category_urls))

        status.success("Traitement terminé.")
        heartbeat.success(f"✅ Terminé – {datetime.now().strftime('%H:%M:%S')}")

        # ===================================================
        # EXPORT SHEETS
        # ===================================================
        if export_sheets:
            if sheets_error:
                st.error(f"Export Sheets impossible : {sheets_error}")
                st.warning("Vérifie que le tableur est partagé avec l’email du service account.")
            else:
                try:
                    # ✅ AJOUT heartbeat (export)
                    heartbeat.info(f"📤 Export Google Sheets… {datetime.now().strftime('%H:%M:%S')}")

                    summary_rows = []
                    all_suspects_rows = []
                    all_products_rows = []

                    for item in st.session_state.results_by_category:
                        cat_url = item["category_url"]
                        title = item["category_title"] or get_category_slug(cat_url) or "Categorie"
                        tab_title = sanitize_sheet_title(title)

                        df_all = item["df"].copy()
                        if df_all.empty:
                            df_all = pd.DataFrame({"info": [f"Aucun produit (ou tous exclus) pour: {cat_url}"]})
                        else:
                            # enrich global ALL_PRODUCTS
                            df_all.insert(0, "category_url", cat_url)
                            df_all.insert(1, "category_title", item["category_title"])
                            df_all.insert(2, "mot_cle_principal", item["category_main_term"])

                            all_products_rows.append(df_all)

                        # onglet catégorie
                        write_df_to_sheet(sh, tab_title, df_all)

                        # suspects
                        if "suspect" in df_all.columns:
                            df_sus = df_all[df_all["suspect"] == True].copy()
                        else:
                            df_sus = pd.DataFrame()

                        if not df_sus.empty:
                            all_suspects_rows.append(df_sus)

                        # summary
                        n_total = int(len(df_all)) if "url" in df_all.columns else 0
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

                        # mise à jour exclusions
                        update_exclusions_from_df(sh, cat_url, item["df"])

                    df_summary = pd.DataFrame(summary_rows)
                    write_df_to_sheet(sh, "SUMMARY", df_summary)

                    if all_suspects_rows:
                        df_sus_all = pd.concat(all_suspects_rows, ignore_index=True)
                        write_df_to_sheet(sh, "SUSPECTS", df_sus_all)
                    else:
                        write_df_to_sheet(sh, "SUSPECTS", pd.DataFrame({"info": ["Aucun suspect"]}))

                    # ✅ ALL_PRODUCTS (tout le monde, au cas où)
                    if all_products_rows:
                        df_all_products = pd.concat(all_products_rows, ignore_index=True)

                        # colonnes “nécessaires” en priorité
                        preferred = [
                            "category_url", "category_title", "mot_cle_principal",
                            "nom", "marque", "prix", "prix_num",
                            "anomalie_score", "niveau_anomalie", "suspect",
                            "score_fiche_technique", "ano_fiche_technique",
                            "score_description", "ano_description",
                            "raison_suspecte", "reco_action",
                            "decision_client", "exclure_prochaine_analyse",
                            "category_title_keywords", "desc_kw_coverage",
                            "url", "image_url",
                            "similarite_moyenne", "similarite_categorie",
                            "similarite_image_moyenne", "similarite_forme_moyenne",
                            "similarite_couleur_moyenne", "similarite_image_globale_moyenne",
                            "fiche_tech_issues", "nb_fiche_champs_consideres",
                        ]
                        cols = [c for c in preferred if c in df_all_products.columns] + [c for c in df_all_products.columns if c not in preferred]
                        df_all_products = df_all_products[cols]

                        write_df_to_sheet(sh, "ALL_PRODUCTS", df_all_products)
                    else:
                        write_df_to_sheet(sh, "ALL_PRODUCTS", pd.DataFrame({"info": ["Aucun produit"]}))

                    st.success("Export terminé (onglet par catégorie + SUMMARY + SUSPECTS + ALL_PRODUCTS + EXCLUSIONS).")
                    heartbeat.success(f"✅ Export terminé – {datetime.now().strftime('%H:%M:%S')}")

                except Exception as e:
                    st.error(f"Export Sheets impossible : {e}")
                    st.warning("Vérifie que le tableur est partagé avec l’email du service account.")


# ===================================================
# AFFICHAGE RESULTATS
# ===================================================

if st.session_state.results_by_category:
    st.subheader("📦 Résultats par catégorie")

    dfs = []
    for item in st.session_state.results_by_category:
        df = item["df"].copy()
        if not df.empty:
            df.insert(0, "category_url", item["category_url"])
            df.insert(1, "category_title", item["category_title"])
            dfs.append(df)

    if dfs:
        df_global = pd.concat(dfs, ignore_index=True)
        csv_global = df_global.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger le CSV (toutes catégories)",
            csv_global,
            file_name="produits_sfactory_toutes_categories.csv",
            mime="text/csv",
        )

    for item in st.session_state.results_by_category:
        cat_url = item["category_url"]
        title = item["category_title"] or cat_url
        df_final = item["df"]

        with st.expander(f"📌 {title}"):
            st.write(f"**URL catégorie :** {cat_url}")
            st.write(f"**Mot-clé principal déduit :** `{item['category_main_term']}`")
            if item["category_intro"]:
                st.write("**Intro catégorie (utilisée pour similarité_categorie)**")
                st.write(item["category_intro"])

            if df_final.empty:
                st.info("Aucune donnée (catégorie vide, inaccessible, ou tout a été exclu).")
                continue

            st.dataframe(df_final, use_container_width=True)

            csv = df_final.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Télécharger le CSV ({sanitize_sheet_title(item['category_title'] or get_category_slug(cat_url))})",
                csv,
                file_name=f"produits_{get_category_slug(cat_url) or 'categorie'}.csv",
                mime="text/csv",
            )

            if "anomalie_score" in df_final.columns:
                cols = [
                    "nom", "sku", "marque", "prix", "prix_num",
                    "title_qty", "title_size_raw", "title_size_value", "title_size_unit",
                    "similarite_categorie",
                    "similarite_image_moyenne", "similarite_forme_moyenne",
                    "similarite_couleur_moyenne", "similarite_image_globale_moyenne",
                    "category_title_keywords", "desc_kw_coverage",
                    "score_fiche_technique", "ano_fiche_technique", "fiche_tech_issues",
                    "score_description", "ano_description",
                    "desc_img_urls", "desc_img_ok", "desc_img_reason", "desc_img_sim_to_main",
                    "anomalie_score", "niveau_anomalie", "suspect",
                    "raison_suspecte", "reco_action",
                    "decision_client", "exclure_prochaine_analyse",
                    "mots_cles_principaux",
                    "url",
                ]
                cols = [c for c in cols if c in df_final.columns]

                st.markdown("#### ⚠️ Produits à traiter en priorité")
                suspects = df_final[df_final.get("suspect", False) == True].sort_values("anomalie_score", ascending=False)
                if not suspects.empty:
                    st.dataframe(suspects[cols], use_container_width=True)
                else:
                    st.caption("Aucun suspect pour cette catégorie.")
