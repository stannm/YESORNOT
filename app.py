# app.py
# Streamlit chatbot to assess basic laser-cut feasibility from a PDF technical plan
# PoC tailored for sheet-metal / laser cutting workflows

import io
import re
import math
from dataclasses import dataclass
from typing import List, Dict, Any

import streamlit as st

# Optional heavy deps are imported lazily
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

__VERSION__ = "0.4.1"

st.set_page_config(page_title="Vérif PDF – Laser", page_icon="🧪", layout="wide")

# CSS pour effet zoom et néons rouges au survol
def inject_css():
    st.markdown(
        """
        <style>
        .zoomable:hover {
            transform: scale(1.2);
            box-shadow: 0 0 15px red;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_css()

st.title("🤖 Chatbot faisabilité – Plans PDF (découpe laser)")
st.caption("PoC – Analyse basique des plans en PDF pour la découpe laser · v" + __VERSION__)

# --- Styles globaux (hover zoom + néon rouge) ---
st.markdown(
    """
    <style>
    .hover-zoom {transition: transform .18s ease, box-shadow .18s ease; border-radius: 12px;}
    .hover-zoom:hover {transform: scale(1.03); box-shadow: 0 0 12px 2px rgba(255,0,0,.7);} 
    .metric-card {padding:10px;border:1px solid rgba(255,0,0,.25);border-radius:12px;margin-bottom:8px;}
    .neon-red {box-shadow: 0 0 0 rgba(0,0,0,0);} 
    .neon-red:hover {box-shadow: 0 0 10px #ff0033, 0 0 20px #ff0033, 0 0 30px #ff0033;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Paramètres machine / procédé")
    bed_w = st.number_input("Largeur utile (mm)", 100, 4000, 1500, help="Largeur utile de la table de découpe.")
    bed_h = st.number_input("Longueur utile (mm)", 100, 4000, 3000, help="Longueur utile de la table de découpe.")

    st.subheader("Technologie")
    tech = st.selectbox("Procédé", ["Fibre", "CO₂", "Plasma", "Jet d'eau"], index=0)

    st.subheader("Contraintes mini (règles) – Métal")
    thickness = st.number_input("Épaisseur matière (mm)", 0.1, 50.0, 3.0, step=0.1)
    kerf = st.number_input("Kerf estimé (mm)", 0.05, 2.0, 0.15, step=0.01, help="Largeur de trait de coupe")
    min_web = st.number_input("Âme minimale entre coupes (mm)", 0.1, 10.0, 0.7, step=0.1)
    min_hole = st.number_input("Ø mini perçage/découpe (mm)", 0.3, 20.0, 1.2, step=0.1)
    min_inner_radius = st.number_input("Rayon intérieur mini (mm)", 0.2, 20.0, max(0.5, round(thickness*0.7,2)), step=0.1)

    st.subheader("Checks divers")
    allow_image_only = st.checkbox("Accepter PDF image (scan)", value=False)

st.markdown("---")

st.write("**Charge un plan PDF** (idéalement provenant d'un DXF/STEP exporté) puis pose ta question au chatbot.")

uploaded = st.file_uploader("PDF du plan", type=["pdf"]) 

@dataclass
class PDFAnalysis:
    ok: bool
    messages: List[str]
    page_sizes_mm: List[Dict[str, float]]
    has_vectors: bool
    has_text: bool
    bbox_ok: bool
    perimeter_mm: float = 0.0


def mm_from_pt(pt):
    return pt * 25.4 / 72.0


def compute_perimeter(drawings) -> float:
    """Approximate perimeter from vector paths (in mm)."""
    total = 0.0
    for d in drawings:
        for item in d["items"]:
            if item[0] == "l":  # line segment
                (x0, y0), (x1, y1) = item[1], item[2]
                length_pt = math.dist((x0, y0), (x1, y1))
                total += mm_from_pt(length_pt)
    return total


def analyze_pdf(file_bytes: bytes) -> PDFAnalysis:
    messages = []
    if not fitz:
        return PDFAnalysis(False, ["PyMuPDF (fitz) non installé."], [], False, False, False)

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return PDFAnalysis(False, [f"Erreur d'ouverture PDF: {e}"], [], False, False, False)

    page_sizes_mm = []
    has_vectors = False
    has_text = False
    bbox_ok = True

    for i, page in enumerate(doc):
        rect = page.rect  # points
        w_mm = mm_from_pt(rect.width)
        h_mm = mm_from_pt(rect.height)
        page_sizes_mm.append({"page": i+1, "w_mm": w_mm, "h_mm": h_mm})

        # Detect drawings (vector paths)
        try:
            draws = page.get_drawings()
            if draws:
                has_vectors = True
        except Exception:
            draws = []

        # Detect text
        try:
            text = page.get_text("text") or ""
            if text.strip():
                has_text = True
        except Exception:
            pass

        # Table size vs bed (toujours en mm)
        if not ((w_mm <= bed_w and h_mm <= bed_h) or (h_mm <= bed_w and w_mm <= bed_h)):
            bbox_ok = False

    if not has_vectors and not allow_image_only:
        messages.append("Le PDF ne contient pas de tracés vectoriels détectables (probable scan). Fournir un export vectoriel (DXF/SVG/PDF vectoriel).")

    if has_text:
        messages.append("Texte détecté – je peux tenter d'extraire cotes / annotations (mm).")
    else:
        messages.append("Aucun texte détecté.")

    if not bbox_ok:
        messages.append(f"Taille page incompatible avec la table ({bed_w}×{bed_h} mm). Envisager un échelle/tiling.")

    ok = has_vectors or allow_image_only
    return PDFAnalysis(ok=ok, messages=messages, page_sizes_mm=page_sizes_mm, has_vectors=has_vectors, has_text=has_text, bbox_ok=bbox_ok) non installé."], [], False, False, False)

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        return PDFAnalysis(False, [f"Erreur d'ouverture PDF: {e}"], [], False, False, False)

    page_sizes_mm = []
    has_vectors = False
    has_text = False
    bbox_ok = True
    total_perimeter = 0.0

    for i, page in enumerate(doc):
        rect = page.rect  # points
        w_mm = mm_from_pt(rect.width)
        h_mm = mm_from_pt(rect.height)
        page_sizes_mm.append({"page": i+1, "w_mm": w_mm, "h_mm": h_mm})

        # Detect drawings (vector paths)
        try:
            draws = page.get_drawings()
            if draws:
                has_vectors = True
                total_perimeter += compute_perimeter(draws)
        except Exception:
            pass

        # Detect text
        try:
            text = page.get_text("text") or ""
            if text.strip():
                has_text = True
        except Exception:
            pass

        # Table size vs bed
        if not ((w_mm <= bed_w and h_mm <= bed_h) or (h_mm <= bed_w and w_mm <= bed_h)):
            bbox_ok = False

    if not has_vectors and not allow_image_only:
        messages.append("Le PDF ne contient pas de tracés vectoriels détectables (probable scan). Fournir un export vectoriel (DXF/SVG/PDF vectoriel).")

    if has_text:
        messages.append("Texte détecté – je peux tenter d'extraire cotes / annotations.")
    else:
        messages.append("Aucun texte détecté.")

    if not bbox_ok:
        messages.append(f"Taille page incompatible avec la table ({bed_w}×{bed_h} mm). Envisager un échelle/tiling.")

    ok = has_vectors or allow_image_only
    return PDFAnalysis(ok=ok, messages=messages, page_sizes_mm=page_sizes_mm, has_vectors=has_vectors, has_text=has_text, bbox_ok=bbox_ok, perimeter_mm=total_perimeter)


# Simple rule engine (text heuristics)
DIM_RE = re.compile(r"(Ø\s*\d+[\.,]?\d*|R\s*\d+[\.,]?\d*|\d+[\.,]?\d*\s*mm)")


def extract_numeric_values_as_mm(text: str) -> List[float]:
    vals = []
    for token in re.findall(r"\d+[\.,]?\d*", text):
        try:
            vals.append(float(token.replace(",", ".")))
        except Exception:
            pass
    return vals


def run_rules_on_text(text: str) -> Dict[str, Any]:
    findings = []
    dims = DIM_RE.findall(text)
    nums = extract_numeric_values_as_mm(text)

    if dims:
        findings.append(f"Repères dimensionnels repérés: {', '.join(dims[:10])}{' …' if len(dims)>10 else ''}")

    # Heuristic: look for very petits Ø (< min_hole)
    small_values = [v for v in nums if v < min_hole]
    if small_values:
        findings.append(f"⚠️ Valeurs inférieures au Ø mini ({min_hole} mm) détectées: ex. {small_values[:5]}")

    # Heuristic for rayon intérieur
    inner_r = [v for v in nums if v < min_inner_radius]
    if inner_r:
        findings.append(f"⚠️ Rayons < rayon intérieur mini ({min_inner_radius} mm) détectés: ex. {inner_r[:5]}")

    return {"findings": findings}


# --- Utilitaires géométrie (périmètre à plat des contours fermés) ---
from math import hypot

def _dist(p0, p1):
    return hypot(p1[0]-p0[0], p1[1]-p0[1])

# Approche générique pour PyMuPDF: on parcourt page.get_drawings() et on somme les longueurs.
# On estime les courbes de Bézier par échantillonnage.

def _bezier_len(p0, p1, p2, p3, samples=16):
    # De Casteljau sampling
    def interp(a,b,t):
        return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t)
    def bez(t):
        ab = interp(p0,p1,t); bc = interp(p1,p2,t); cd = interp(p2,p3,t)
        ab2 = interp(ab,bc,t); bc2 = interp(bc,cd,t)
        return interp(ab2,bc2,t)
    pts = [bez(i/samples) for i in range(samples+1)]
    return sum(_dist(pts[i], pts[i+1]) for i in range(samples))


def compute_page_perimeters_mm(page) -> Dict[str, Any]:
    """Retourne périmètres en mm: somme des segments, somme des contours fermés, plus grand contour.
    Limites: PDF doit être vectoriel; identification des contours via proximité start-end.
    """
    total_len_pt = 0.0
    closed_loops_len_pt = []

    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []

    for d in drawings:
        # Chaque d["items"] est une séquence d'éléments: (op, points, ...)
        items = d.get("items", [])
        path_len = 0.0
        start = None
        last = None

        for it in items:
            if not it:
                continue
            op = it[0]
            if op == "l":
                # line: it[1] = (x0,y0,x1,y1)
                x0,y0,x1,y1 = it[1]
                path_len += _dist((x0,y0),(x1,y1))
                if start is None:
                    start = (x0,y0)
                last = (x1,y1)
            elif op == "c":
                # cubic bezier: it[1] = (x0,y0,x1,y1,x2,y2,x3,y3)
                x0,y0,x1,y1,x2,y2,x3,y3 = it[1]
                path_len += _bezier_len((x0,y0),(x1,y1),(x2,y2),(x3,y3))
                if start is None:
                    start = (x0,y0)
                last = (x3,y3)
            elif op == "re":
                # rectangle: it[1] = (x,y,w,h)
                x,y,w,h = it[1]
                path_len += 2.0*(abs(w)+abs(h))
                if start is None:
                    start = (x,y)
                last = (x,y)
            elif op == "m":
                # move to (nouveau sous-chemin): clôture l'actuel
                if start is not None and last is not None:
                    if _dist(start,last) < 1e-3:
                        closed_loops_len_pt.append(path_len)
                    total_len_pt += path_len
                path_len = 0.0
                start = (it[1][0], it[1][1])
                last = start
            elif op == "h":
                # close path
                if start is not None and last is not None:
                    path_len += _dist(last, start)
                    closed_loops_len_pt.append(path_len)
                    total_len_pt += path_len
                    path_len = 0.0
                    start = None
                    last = None
        # fin items -> accumulate
        if path_len > 0.0:
            total_len_pt += path_len
            if start is not None and last is not None and _dist(start,last) < 1e-3:
                closed_loops_len_pt.append(path_len)

    # Conversion points ➜ mm
    total_len_mm = mm_from_pt(total_len_pt)
    closed_loops_mm = [mm_from_pt(v) for v in closed_loops_len_pt]
    largest_loop_mm = max(closed_loops_mm) if closed_loops_mm else 0.0

    return {
        "total_path_mm": total_len_mm,
        "sum_closed_mm": sum(closed_loops_mm),
        "largest_closed_mm": largest_loop_mm,
        "n_closed": len(closed_loops_mm),
    }

# --- Chat Section ---
st.subheader("💬 Chat")
user_q = st.text_input("Pose ta question (ex: \"Est-ce faisable en 3 mm acier ?\")")

analysis = None
text_dump = ""

if uploaded:
    file_bytes = uploaded.read()
    analysis = analyze_pdf(file_bytes)

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown('<div class="metric-card hover-zoom neon-red">', unsafe_allow_html=True)
        st.metric("Pages", len(analysis.page_sizes_mm))
        st.markdown('</div>', unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="metric-card hover-zoom neon-red">', unsafe_allow_html=True)
        st.metric("Vecteurs détectés", "Oui" if analysis.has_vectors else "Non")
        st.markdown('</div>', unsafe_allow_html=True)
    with colC:
        st.markdown('<div class="metric-card hover-zoom neon-red">', unsafe_allow_html=True)
        st.metric("Texte détecté", "Oui" if analysis.has_text else "Non")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("**Formats page (mm)**")
    st.dataframe({"page": [p["page"] for p in analysis.page_sizes_mm],
                  "largeur_mm": [round(p["w_mm"],1) for p in analysis.page_sizes_mm],
                  "hauteur_mm": [round(p["h_mm"],1) for p in analysis.page_sizes_mm]})

    # --- Périmètres (mm) ---
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_perims = [compute_page_perimeters_mm(pg) for pg in doc]
    except Exception:
        page_perims = []

    if page_perims:
        st.subheader("📏 Périmètre à plat (mm) – Estimation")
        for i, data in enumerate(page_perims, start=1):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card hover-zoom neon-red">', unsafe_allow_html=True)
                st.metric(f"Page {i}", "", help="Résultats pour cette page")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card hover-zoom neon-red">', unsafe_allow_html=True)
                st.metric("Total tracés (mm)", f"{data['total_path_mm']:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card hover-zoom neon-red">', unsafe_allow_html=True)
                st.metric("Σ contours fermés (mm)", f"{data['sum_closed_mm']:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-card hover-zoom neon-red">', unsafe_allow_html=True)
                st.metric("Plus grand contour (mm)", f"{data['largest_closed_mm']:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)

    if fitz and analysis.has_text:
        try:
            # concat text for all pages
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text_dump = "
".join((pg.get_text("text") or "").strip() for pg in doc)
            with st.expander("Voir texte extrait"):
                st.code(text_dump[:5000] + ("
…" if len(text_dump) > 5000 else ""))
        except Exception:
            pass

    st.info("
".join(analysis.messages))

    # Exemple visuel avec hover + néons rouges
    st.markdown('<div class="zoomable">🟥 Aperçu interactif du plan (placeholder)</div>', unsafe_allow_html=True)

if st.button("Analyser la faisabilité", disabled=(uploaded is None)):
    if not analysis:
        st.warning("Charge d'abord un PDF.")
    else:
        verdict_msgs = []
        score = 100

        if not analysis.bbox_ok:
            score -= 25
            verdict_msgs.append("Le format du plan dépasse la table – prévoir mise à l'échelle ou découpe en plusieurs panneaux.")
        if not analysis.has_vectors and not allow_image_only:
            score -= 40
            verdict_msgs.append("PDF non vectoriel – export DXF/SVG requis pour une vraie faisabilité.")
        if kerf > 0.25 and tech == "Fibre":
            score -= 5
            verdict_msgs.append("Kerf élevé pour fibre – vérifie la focal/paramètres.")
        if min_web < kerf * 3:
            score -= 10
            verdict_msgs.append("Âmes trop fines vs kerf – risque de fusion/fragilité.")
        if thickness > 6 and tech == "CO₂":
            score -= 10
            verdict_msgs.append("CO₂ sur >6 mm acier : limites possibles de perçage/qualité.")

        text_findings = []
        if text_dump:
            res = run_rules_on_text(text_dump)
            text_findings = res["findings"]
            if any(f.startswith("⚠️") for f in text_findings):
                score -= 10

        score = max(0, min(100, score))
        status = "✅ Faisable (PoC)" if score >= 70 else ("🟨 À valider" if score >= 45 else "❌ Non conforme (à ce stade)")

        st.subheader("Résultat")
        st.metric("Score faisabilité (PoC)", f"{score}/100", help="Score heuristique – à affiner avec des règles métier.")
        st.success(status) if score >= 70 else st.warning(status) if score >= 45 else st.error(status)

        st.write("### Points clés")
        for m in verdict_msgs:
            st.write("- " + m)
        for f in text_findings:
            st.write("- " + f)

        st.caption("⚠️ PoC – Ne remplace pas une analyse CAO (DXF/STEP) réelle ni la vérification par un chargé d'affaires.")

st.markdown("---")

st.write("### Intégration chatbot (LLM) – optionnel")
st.markdown(
"""
- Tu peux brancher une API LLM (OpenAI, etc.) pour répondre en langage naturel.
- L'idée : passer au LLM le *résumé d'analyse* + les *paramètres machine* + la *question utilisateur*,
  avec des **outils** (functions) permettant d'appeler `analyze_pdf` et de renvoyer un verdict structuré.
- **Périmètre** : la V1 calcule des *périmètres en mm* à partir des **contours fermés** détectés (PDF vectoriel requis).
  Pour une pièce *pliée*, si le PDF n'est pas une **mise à plat**, il faudra le **DXF développé** (K‑factor / BA) pour un résultat fiable.
"""
)

st.write("### Roadmap technique (à implémenter ensuite)")
st.markdown(
"""
1. **Extraction vectorielle fiable** : intégration d'un parseur DXF/SVG (ex. `ezdxf`) pour reconstruire les polylignes/arc/cercles et fermer les boucles proprement.
2. **Détection mise à plat vs. vue pliée** : heuristiques (cartouche *FLAT PATTERN*, calques, repères de pli `V-`/`R-`).
3. **Dépliage (si 3D)** : si STEP/Inventor dispo ➜ calcul BA/BD via K‑factor; sinon exiger la mise à plat fournie.
4. **Aperçu interactif** : rendu SVG/Canvas des contours avec **zoom au survol + halo néon rouge** élément par élément.
5. **Rapport** : export PDF/CSV des périmètres (total, Σ fermés, plus grand contour) et des alertes (mm).
"""
)

st.caption("© PoC éducatif – adapté à la découpe laser tôle. Parfait pour une v1 hébergée sur Streamlit Cloud.")("© PoC éducatif – adapté à la découpe laser tôle. Parfait pour une v1 hébergée sur Streamlit Cloud.")
