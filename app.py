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

__VERSION__ = "0.3.0"

st.set_page_config(page_title="Vérif PDF – Laser", page_icon="🧪", layout="wide")
st.title("🤖 Chatbot faisabilité – Plans PDF (découpe laser)")
st.caption("PoC – Analyse basique des plans en PDF pour la découpe laser · v" + __VERSION__)

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


def mm_from_pt(pt):
    return pt * 25.4 / 72.0


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
    return PDFAnalysis(ok=ok, messages=messages, page_sizes_mm=page_sizes_mm, has_vectors=has_vectors, has_text=has_text, bbox_ok=bbox_ok)


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
        st.metric("Pages", len(analysis.page_sizes_mm))
    with colB:
        st.metric("Vecteurs détectés", "Oui" if analysis.has_vectors else "Non")
    with colC:
        st.metric("Texte détecté", "Oui" if analysis.has_text else "Non")

    st.write("**Formats page (mm)**")
    st.dataframe({"page": [p["page"] for p in analysis.page_sizes_mm],
                  "largeur_mm": [round(p["w_mm"],1) for p in analysis.page_sizes_mm],
                  "hauteur_mm": [round(p["h_mm"],1) for p in analysis.page_sizes_mm]})

    if fitz and analysis.has_text:
        try:
            # concat text for all pages
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text_dump = "\n".join((pg.get_text("text") or "").strip() for pg in doc)
            with st.expander("Voir texte extrait"):
                st.code(text_dump[:5000] + ("\n…" if len(text_dump) > 5000 else ""))
        except Exception:
            pass

    st.info("\n".join(analysis.messages))

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

        # Text-based checks
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
- Pour un usage pro : ajouter un **parseur vectoriel** (DXF/SVG) pour vérifier :
  - fermetures de contours, intersections, tolérance min. entre traits (`min_web`)
  - Ø mini, rayons intérieurs vs épaisseur (R ≥ 0.7×épaisseur typ.)
  - pièces > format tôle / > format machine
  - détection d'échelle / unités (mm), blocs titre, matière, ISO 2768, etc.
"""
)

st.write("### Roadmap technique (à implémenter ensuite)")
st.markdown(
"""
1. **Extraction vectorielle** :
   - Si PDF vectoriel : `page.get_drawings()` ➜ segments/bez., aire, longueurs (post-tri en mm).
   - Sinon : récupération du **DXF** original (meilleure source vérité).
2. **Géométrie** :
   - Construire des graphes de contours, détecter **zones fermées**, **auto-intersections** (via `shapely`).
3. **Règles métier** :
   - Kerf, âme mini (≥ 3×kerf), Ø mini (≥ 1.2×épaisseur acier), Rint (≥ 0.7×ép.), ponts d'attache, marquage.
4. **Rapport PDF** :
   - Générer un rapport exportable (OK/KO + détails) + image d'aperçu.
5. **Persistance** :
   - Logs & fichiers sur **Supabase** (Postgres + Storage) pour l'historique et l'admin.
"""
)

st.caption("© PoC éducatif – adapté à la découpe laser tôle. Parfait pour une v1 hébergée sur Streamlit Cloud.")
