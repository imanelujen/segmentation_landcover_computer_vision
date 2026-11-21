from io import BytesIO
from pathlib import Path
import sys

import streamlit as st
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "unet_best.pth"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import CLASS_COLORS, CLASS_NAMES, load_model, predict_image

st.set_page_config(
    page_title="Segmentation LandCover.ai",
    page_icon="🛰️",
    layout="wide",
)


@st.cache_resource(show_spinner="Chargement du modèle U-Net…")
def get_model():
    return load_model(CHECKPOINT_PATH)


def main():
    st.title("🛰️ Segmentation d’images aériennes")
    st.write(
        "Téléversez une image RGB pour obtenir une carte thématique "
        "de l’occupation du sol."
    )

    with st.sidebar:
        st.header("Classes du modèle")
        for class_name, color in zip(CLASS_NAMES, CLASS_COLORS):
            color_hex = "#" + "".join(f"{int(channel):02X}" for channel in color)
            st.markdown(
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background:{color_hex};border:1px solid #999;margin-right:8px;'></span>"
                f"{class_name}",
                unsafe_allow_html=True,
            )
        st.caption("Le modèle prédit uniquement ces cinq classes.")

    uploaded_file = st.file_uploader(
        "Choisir une image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Image aérienne RGB. Les grandes images sont traitées par tuiles de 512 × 512 pixels.",
    )

    if uploaded_file is None:
        st.info("Ajoutez une image PNG, JPG ou TIFF pour commencer.")
        st.warning(
            "Les prédictions sont automatiques. Le modèle ne distingue pas "
            "l’agriculture ni le sol nu; les résultats peuvent varier selon la région."
        )
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"Impossible d’ouvrir cette image : {error}")
        return

    st.caption(f"Image : {image.width} × {image.height} pixels")

    if st.button("Lancer la segmentation", type="primary", use_container_width=True):
        if not CHECKPOINT_PATH.is_file():
            st.error(f"Checkpoint introuvable : {CHECKPOINT_PATH}")
            return

        try:
            model, device = get_model()
            with st.spinner(f"Segmentation en cours sur {device}…"):
                result = predict_image(image, model, device)
        except Exception as error:
            st.exception(error)
            return

        st.success("Segmentation terminée.")
        original_col, mask_col, overlay_col = st.columns(3)
        original_col.image(result["original"], caption="Image originale", use_container_width=True)
        mask_col.image(result["colored_mask"], caption="Carte thématique", use_container_width=True)
        overlay_col.image(result["overlay"], caption="Superposition", use_container_width=True)

        st.subheader("Répartition des classes prédites")
        percentages = result["class_percentages"]
        metric_columns = st.columns(len(percentages))
        for column, (name, percentage) in zip(metric_columns, percentages.items()):
            column.metric(name, f"{percentage:.2f} %")

        mask_buffer = BytesIO()
        Image.fromarray(result["colored_mask"]).save(mask_buffer, format="PNG")
        st.download_button(
            "Télécharger le masque coloré",
            data=mask_buffer.getvalue(),
            file_name="carte_thematique.png",
            mime="image/png",
        )
        st.caption(
            "Les pourcentages décrivent les pixels prédits et ne constituent "
            "pas une mesure de la précision sur cette image."
        )


if __name__ == "__main__":
    main()
