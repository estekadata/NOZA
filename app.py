import os
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

from noza.config import load_config
from noza.cache import DiskCache
from noza.database import AnalysisDB
from noza.scraper import (
    get_soup, extract_category_title, extract_category_intro_text,
    build_category_main_term, get_product_urls, scrape_products_parallel,
    scrape_product, get_category_slug,
)
from noza.analysis import (
    enrich_structured_features, compute_similarity, compute_fiche_technique_incoherence,
    add_outlier_flags_and_reasons, add_client_decision_columns,
    compute_common_title_terms, description_keyword_coverage,
)
from noza.image import compute_image_similarity_optimized
from noza.sheets import (
    HAS_SHEETS, sheets_client_from_secrets, sheets_client_from_file,
    extract_sheet_id, sanitize_sheet_title, write_df_to_sheet,
    ensure_exclusions_sheet, load_excluded_urls, update_exclusions_from_df,
    load_decisions_map_from_all_products, delete_all_tabs_except,
)
from noza.charts import (
    HAS_PLOTLY, score_distribution_chart, price_boxplot, anomaly_level_pie,
    score_vs_price_scatter, history_trend_chart, category_comparison_bar,
)
from noza.logger import log

# ===================================================
# STREAMLIT CONFIG & CUSTOM CSS
# ===================================================

st.set_page_config(
    page_title="noza",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');
    :root {
        --bg: #f8f8fa;
        --surface: #ffffff;
        --surface-raised: #f3f3f6;
        --border: #e4e4e8;
        --border-hover: #c8c8d0;
        --text: #1a1a2e;
        --text-secondary: #5a5a72;
        --text-muted: #9494a8;
        --accent: #6d5cff;
        --accent-hover: #5a48e6;
        --accent-bg: rgba(109, 92, 255, 0.06);
        --accent-glow: rgba(109, 92, 255, 0.12);
        --accent-light: #eeebff;
        --success: #0d9f6e;
        --success-bg: #ecfdf5;
        --warning: #d97706;
        --warning-bg: #fffbeb;
        --danger: #dc2626;
        --danger-bg: #fef2f2;
    }
    * { color-scheme: light; }
    .stApp {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--bg) !important;
        color: var(--text);
    }

    /* --- Header --- */
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(79, 70, 229, 0.15);
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -40%; right: -10%;
        width: 350px; height: 350px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    .main-header::after {
        content: '';
        position: absolute;
        bottom: -30%; left: 10%;
        width: 250px; height: 250px;
        background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        position: relative;
    }
    .main-header .brand-dot {
        display: inline-block;
        width: 10px; height: 10px;
        background: #fbbf24;
        border-radius: 50%;
        margin-left: 4px;
        vertical-align: middle;
        box-shadow: 0 0 10px rgba(251, 191, 36, 0.4);
    }
    .main-header p {
        color: rgba(255,255,255,0.75);
        font-size: 0.9rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
        font-weight: 400;
        position: relative;
    }

    /* --- Stat cards --- */
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stat-card:hover {
        border-color: var(--accent);
        box-shadow: 0 4px 16px var(--accent-glow);
        transform: translateY(-2px);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1;
        letter-spacing: -0.5px;
    }
    .stat-label {
        color: var(--text-muted);
        font-size: 0.7rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }

    /* --- Progress --- */
    .progress-container {
        background: var(--surface);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        border: 1px solid var(--border);
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .progress-title {
        font-weight: 500;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .status-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .status-dot.running {
        background: var(--accent);
        animation: dotPulse 1.5s ease-in-out infinite;
    }
    .status-dot.success { background: var(--success); }
    .status-dot.warning { background: var(--warning); }
    .status-dot.error { background: var(--danger); }
    @keyframes dotPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .status-text {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: var(--text) !important; }
    section[data-testid="stSidebar"] label {
        color: var(--text-secondary) !important;
        font-weight: 500;
        font-size: 0.8rem;
    }
    section[data-testid="stSidebar"] .stTextInput > div > div > input,
    section[data-testid="stSidebar"] .stTextArea > div > div > textarea {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text) !important;
        font-size: 0.85rem;
    }
    section[data-testid="stSidebar"] .stTextInput > div > div > input:focus,
    section[data-testid="stSidebar"] .stTextArea > div > div > textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-bg);
    }

    /* --- Buttons --- */
    .stButton > button {
        background: var(--accent);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 4px rgba(109, 92, 255, 0.2);
    }
    .stButton > button:hover {
        background: var(--accent-hover);
        box-shadow: 0 4px 16px rgba(109, 92, 255, 0.3);
        transform: translateY(-1px);
    }

    /* --- Download buttons --- */
    .stDownloadButton > button {
        background: var(--surface) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px;
        font-weight: 500;
        box-shadow: none !important;
    }
    .stDownloadButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: var(--accent-bg) !important;
    }

    /* --- Expander --- */
    .streamlit-expanderHeader {
        background: var(--surface);
        border-radius: 10px;
        border: 1px solid var(--border);
        font-weight: 500;
        font-size: 0.85rem;
    }

    /* --- Data --- */
    .stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }
    div[data-testid="metric-container"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--border);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.7rem 1.25rem;
        color: var(--text-muted);
        border-bottom: 2px solid transparent;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--text-secondary); }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom-color: var(--accent) !important;
    }

    /* --- Alerts --- */
    .stAlert { border-radius: 10px; }

    /* --- General --- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: var(--bg) !important;
    }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-hover); }
    hr { border-color: var(--border) !important; }
    .block-container { padding-top: 2rem; }
    h2 { color: var(--text) !important; font-weight: 700; font-size: 1.35rem; letter-spacing: -0.3px; }
    h3 { color: var(--text) !important; font-weight: 600; font-size: 1.05rem; }
    h4 { color: var(--text-secondary) !important; font-weight: 600; font-size: 0.95rem; }
    p, li { color: var(--text-secondary); }
    .stButton > button span, .stButton > button p,
    .stDownloadButton > button span, .stDownloadButton > button p { color: inherit !important; }
    code {
        color: var(--accent);
        background: var(--accent-light);
        border-radius: 5px;
        padding: 0.15em 0.4em;
        font-size: 0.85em;
    }
    .stSpinner > div { color: var(--accent) !important; }
    .stProgress > div > div > div { background: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

if "results_by_category" not in st.session_state:
    st.session_state.results_by_category = []

# ===================================================
# LOAD CONFIG
# ===================================================

config_path = os.environ.get("NOZA_CONFIG", None)
config = load_config(config_path)

cache_cfg = config.get("cache", {})
cache = DiskCache(
    cache_dir=cache_cfg.get("directory", ".noza_cache"),
    default_ttl=cache_cfg.get("http_ttl_seconds", 3600),
) if cache_cfg.get("enabled", True) else None

db_cfg = config.get("database", {})
db = AnalysisDB(db_cfg.get("path", "noza_history.db")) if db_cfg.get("enabled", True) else None

base_domain = config.get("base_domain", "https://www.planete-sfactory.com")

# ===================================================
# UI COMPONENTS
# ===================================================

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>noza<span class="brand-dot"></span></h1>
        <p>Analyse de catalogues produits</p>
    </div>
    """, unsafe_allow_html=True)

def render_stats_cards(total_products, total_suspects, categories_count, avg_score):
    cols = st.columns(4)
    stats = [
        (total_products, "Produits"),
        (total_suspects, "Suspects"),
        (categories_count, "Categories"),
        (f"{avg_score:.1f}", "Score moyen"),
    ]
    for col, (value, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

def render_progress_status(status_type, message, details=""):
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-title">
            <span class="status-dot {status_type}"></span>
            <span class="status-text">{message}</span>
        </div>
        {f'<p style="color: var(--text-muted); margin: 0.25rem 0 0 1rem; font-size: 0.8rem;">{details}</p>' if details else ''}
    </div>
    """, unsafe_allow_html=True)

# ===================================================
# SIDEBAR
# ===================================================

render_header()

# Tabs for main content
tab_analyse, tab_history, tab_cache = st.tabs(["Analyse", "Historique", "Cache"])

with st.sidebar:
    st.markdown("### Configuration")

    urls_text = st.text_area(
        "URLs categories (1 par ligne)",
        value="https://www.planete-sfactory.com/vaporisateur/vaporisateur-portable/",
        height=140,
        help="Entrez les URLs des categories a analyser"
    )

    config_file = st.text_input("Fichier config YAML (optionnel)", value="",
                                help="Chemin vers un fichier de config YAML personnalise")
    if config_file and os.path.exists(config_file):
        config = load_config(config_file)
        base_domain = config.get("base_domain", base_domain)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        analyse_image = st.checkbox("Analyse images", value=True)
    with col2:
        parallel_scraping = st.checkbox("Mode parallele", value=True)

    max_pages = st.slider("Max pages/categorie", 1, 200, config.get("scraping", {}).get("max_pages", 80))
    max_workers = st.slider("Workers paralleles", 2, 16,
                            config.get("scraping", {}).get("max_workers", 8)) if parallel_scraping else 1

    st.markdown("---")
    st.markdown("### Export Google Sheets")

    export_sheets = st.checkbox("Activer export Sheets", value=True)

    if export_sheets:
        sheet_url_or_id = st.text_input(
            "URL/ID du tableur",
            value="https://docs.google.com/spreadsheets/d/1lSqjJb-6HZQVfsDVrjeN2wULtFW7luK4d-HmBQOlx58/edit?gid=0#gid=0"
        )
        auth_mode = st.radio("Mode d'authentification", ["Streamlit secrets", "Fichier JSON local"], index=0)
        sa_json_path = ""
        if auth_mode == "Fichier JSON local":
            sa_json_path = st.text_input("Chemin JSON service account")
        wipe_tabs = st.checkbox("Supprimer anciens onglets", value=False)

st.markdown("---")
run = st.button("Lancer l'analyse", use_container_width=True)

progress_placeholder = st.empty()
status_placeholder = st.empty()
metrics_placeholder = st.empty()

# ===================================================
# MAIN ANALYSIS
# ===================================================

if run:
    category_urls = [u.strip() for u in urls_text.splitlines() if u.strip()]

    if not category_urls:
        st.error("Veuillez entrer au moins une URL de categorie.")
    else:
        st.session_state.results_by_category = []

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
                        sa_info = st.secrets.get("gcp_service_account", None)
                        if not sa_info:
                            raise RuntimeError("Secret manquant: st.secrets['gcp_service_account']")
                        client = sheets_client_from_secrets(sa_info)
                    else:
                        if not sa_json_path or not os.path.exists(sa_json_path):
                            raise FileNotFoundError("JSON service account introuvable")
                        client = sheets_client_from_file(sa_json_path)
                    sh = client.open_by_key(spreadsheet_id)
                    ensure_exclusions_sheet(sh)
                    if wipe_tabs:
                        delete_all_tabs_except(sh, keep_titles={"EXCLUSIONS"})
                    decisions_map = load_decisions_map_from_all_products(sh)
                except Exception as e:
                    sheets_error = str(e)

        progress_bar = progress_placeholder.progress(0)
        total_products_scraped = 0
        total_suspects_found = 0

        for idx_cat, category_url in enumerate(category_urls, start=1):
            with status_placeholder.container():
                render_progress_status("running", f"Categorie {idx_cat}/{len(category_urls)}", category_url)

            try:
                soup_cat = get_soup(category_url, cache=cache, config=config)
                category_title = extract_category_title(soup_cat)
                category_intro = extract_category_intro_text(soup_cat, config=config)
                category_main_term = build_category_main_term(category_title, category_url, config)

                product_urls = get_product_urls(category_url, base_domain, cache=cache,
                                                config=config, max_pages=int(max_pages))

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
                    if parallel_scraping:
                        prod_progress = st.progress(0, text="Scraping produits...")

                        def update_progress(completed, total, url):
                            prod_progress.progress(completed / total, text=f"Produit {completed}/{total}")

                        results = scrape_products_parallel(
                            product_urls, base_domain, cache=cache, config=config,
                            max_workers=max_workers, progress_callback=update_progress,
                        )
                        prod_progress.empty()
                    else:
                        results = []
                        for i, prod_url in enumerate(product_urls):
                            try:
                                results.append(scrape_product(prod_url, base_domain, cache, config))
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

                    common_title_terms = compute_common_title_terms(df, top_k=12)
                    df["category_title_keywords"] = ", ".join(common_title_terms)
                    df["desc_kw_coverage"] = df["description"].fillna("").astype(str).apply(
                        lambda d: description_keyword_coverage(d, common_title_terms)
                    )

                    df = enrich_structured_features(df, category_main_term)
                    df = compute_similarity(df, category_intro)

                    if analyse_image:
                        with st.spinner("Analyse des images..."):
                            df = compute_image_similarity_optimized(df, max_workers=max_workers, config=config)

                    df = compute_fiche_technique_incoherence(df, config.get("analysis", {}).get("consensus_ratio", 0.6))
                    df = add_outlier_flags_and_reasons(df, category_main_term, config)
                    df = add_client_decision_columns(df, decisions_map=decisions_map)

                    total_products_scraped += len(df)
                    total_suspects_found += df["suspect"].sum() if "suspect" in df.columns else 0

                # Save to database
                if db and not df.empty:
                    db.save_run(category_url, category_title, df, config)

                st.session_state.results_by_category.append({
                    "category_url": category_url,
                    "category_title": category_title,
                    "category_main_term": category_main_term,
                    "category_intro": category_intro,
                    "df": df
                })

                progress_bar.progress(idx_cat / len(category_urls))

            except Exception as e:
                st.error(f"Erreur categorie {category_url}: {e}")
                st.session_state.results_by_category.append({
                    "category_url": category_url,
                    "category_title": "",
                    "category_main_term": "produit",
                    "category_intro": "",
                    "df": pd.DataFrame()
                })
                progress_bar.progress(idx_cat / len(category_urls))

        with status_placeholder.container():
            render_progress_status("success", "Analyse terminee!",
                                   f"{total_products_scraped} produits analyses, {total_suspects_found} suspects")

        # Google Sheets export
        if export_sheets:
            if sheets_error:
                st.error(f"Export Sheets impossible: {sheets_error}")
            else:
                try:
                    with st.spinner("Export vers Google Sheets..."):
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

                        st.success("Export Google Sheets termine.")

                except Exception as e:
                    st.error(f"Erreur export: {e}")

# ===================================================
# RESULTS DISPLAY
# ===================================================

with tab_analyse:
    if st.session_state.results_by_category:
        st.markdown("## Resultats")

        all_dfs = [item["df"] for item in st.session_state.results_by_category if not item["df"].empty]
        if all_dfs:
            df_combined = pd.concat(all_dfs, ignore_index=True)
            total_products = len(df_combined)
            total_suspects = df_combined["suspect"].sum() if "suspect" in df_combined.columns else 0
            avg_score = df_combined["anomalie_score"].mean() if "anomalie_score" in df_combined.columns else 0

            render_stats_cards(total_products, int(total_suspects),
                              len(st.session_state.results_by_category), avg_score)

            st.markdown("<br>", unsafe_allow_html=True)

            # Charts
            if HAS_PLOTLY:
                st.markdown("### Visualisations")
                chart_cols = st.columns(2)
                with chart_cols[0]:
                    fig = score_distribution_chart(df_combined)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with chart_cols[1]:
                    fig = anomaly_level_pie(df_combined)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                chart_cols2 = st.columns(2)
                with chart_cols2[0]:
                    fig = price_boxplot(df_combined)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with chart_cols2[1]:
                    fig = score_vs_price_scatter(df_combined)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                if len(st.session_state.results_by_category) > 1:
                    fig = category_comparison_bar(st.session_state.results_by_category)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            # CSV export
            csv_global = df_combined.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Telecharger CSV complet",
                csv_global,
                file_name="noza_analyse.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("---")

        for item in st.session_state.results_by_category:
            cat_url = item["category_url"]
            title = item["category_title"] or cat_url
            df_final = item["df"]

            with st.expander(title, expanded=False):
                st.markdown(f"**URL:** `{cat_url}`")
                st.markdown(f"**Mot-cle principal:** `{item['category_main_term']}`")

                if df_final.empty:
                    st.info("Aucune donnee disponible.")
                    continue

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

                if "suspect" in df_final.columns:
                    suspects_df = df_final[df_final["suspect"] == True].sort_values("anomalie_score", ascending=False)
                    if not suspects_df.empty:
                        st.markdown("#### Produits suspects")
                        st.dataframe(suspects_df[display_cols], use_container_width=True)

                csv = df_final.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"Telecharger CSV — {title[:30]}",
                    csv,
                    file_name=f"noza_{get_category_slug(cat_url) or 'categorie'}.csv",
                    mime="text/csv"
                )

# ===================================================
# HISTORY TAB
# ===================================================

with tab_history:
    if db:
        st.markdown("## Historique des analyses")
        db_stats = db.get_stats()
        hcol1, hcol2, hcol3 = st.columns(3)
        with hcol1:
            st.metric("Total analyses", db_stats["total_runs"])
        with hcol2:
            st.metric("Snapshots produits", db_stats["total_product_snapshots"])
        with hcol3:
            st.metric("Categories uniques", db_stats["unique_categories"])

        runs_df = db.get_runs(limit=100)
        if not runs_df.empty:
            if HAS_PLOTLY:
                fig = history_trend_chart(runs_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            st.dataframe(runs_df[["id", "run_date", "category_title", "total_products",
                                   "total_suspects", "avg_anomaly_score"]],
                         use_container_width=True)

            # Run comparison
            st.markdown("### Comparer deux analyses")
            run_ids = runs_df["id"].tolist()
            if len(run_ids) >= 2:
                ccol1, ccol2 = st.columns(2)
                with ccol1:
                    old_run = st.selectbox("Analyse precedente", run_ids, index=1)
                with ccol2:
                    new_run = st.selectbox("Analyse recente", run_ids, index=0)

                if st.button("Comparer"):
                    comparison = db.compare_runs(old_run, new_run)
                    st.markdown(f"**Nouveaux produits:** {len(comparison['added'])}")
                    st.markdown(f"**Produits retires:** {len(comparison['removed'])}")
                    st.markdown(f"**Changements de score:** {len(comparison['score_changes'])}")
                    if comparison["score_changes"]:
                        st.dataframe(pd.DataFrame(comparison["score_changes"]), use_container_width=True)
        else:
            st.info("Aucune analyse enregistree. Lancez une analyse pour commencer.")
    else:
        st.info("Base de donnees desactivee dans la configuration.")

# ===================================================
# CACHE TAB
# ===================================================

with tab_cache:
    st.markdown("## Gestion du cache")
    if cache:
        stats = cache.stats()
        ccol1, ccol2 = st.columns(2)
        with ccol1:
            st.metric("Fichiers en cache", stats["files"])
        with ccol2:
            st.metric("Taille", f"{stats['size_mb']} MB")

        if st.button("Vider le cache"):
            cache.clear()
            st.success("Cache vide!")
            st.rerun()
    else:
        st.info("Cache desactive dans la configuration.")

# ===================================================
# FOOTER
# ===================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-muted); font-size: 0.75rem; padding: 2rem 0 1rem; letter-spacing: 0.05em;">
    noza v0.2
</div>
""", unsafe_allow_html=True)
