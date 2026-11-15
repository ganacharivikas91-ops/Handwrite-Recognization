
import json
from pathlib import Path

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from model.cnn_model import CharacterCNN
from utils.labels import EMNIST_BALANCED_LABELS
from utils.prediction import load_model, predict_image
from utils.preprocessing import ImageInput, load_image


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

PAGES = [
    "Home Dashboard",
    "Draw Character",
    "Upload Image",
    "Prediction Results",
    "Model Performance",
    "About Project",
]


# ==========================================================
# EXISTING MODEL AND PREDICTION FUNCTIONS
# ==========================================================

def load_json(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading trained CNN...")
def get_model():
    return load_model()


def run_prediction(source, original_image, source_name):
    try:
        with st.spinner("AI is analyzing your character..."):
            result = predict_image(source, get_model())

        st.session_state["last_prediction"] = result
        st.session_state["last_input_image"] = original_image
        st.session_state["navigate_to_results"] = True
        st.rerun()

    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Prediction failed: {error}")


# ==========================================================
# UI STYLING
# ==========================================================

def apply_styles():

    st.markdown(
        """
        <style>

        /* MAIN BACKGROUND */

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(
                    circle at 10% 5%,
                    rgba(0, 180, 216, 0.13),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 90% 90%,
                    rgba(45, 100, 190, 0.16),
                    transparent 30%
                ),
                #080E1B;
            color: #F1F5F9;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stMain"] {
            background: transparent;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1450px;
            padding: 2rem 2.2rem 6rem;
            margin: 1.2rem auto;
            background: rgba(15, 25, 43, 0.94);
            border: 1px solid rgba(80, 190, 220, 0.15);
            border-radius: 24px;
            box-shadow: 0 20px 70px rgba(0, 0, 0, 0.35);
        }

        /* SIDEBAR */

        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #0B1424,
                #101D31
            );
            border-right: 1px solid rgba(75, 190, 220, 0.15);
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding: 12px;
            border-radius: 12px;
            transition: all 0.2s ease;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(0, 190, 220, 0.13);
            transform: translateX(4px);
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label p {
            color: #CBD5E1;
            font-weight: 600;
        }

        /* HEADINGS */

        h1, h2, h3 {
            color: #F8FAFC !important;
            letter-spacing: -0.5px;
        }

        .app-kicker {
            color: #38C9DF;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        /* HERO PANEL */

        .hero-panel {
            position: relative;
            overflow: hidden;
            padding: 35px;
            margin: 20px 0 28px;

            background:
                radial-gradient(
                    circle at 90% 10%,
                    rgba(0, 210, 230, 0.20),
                    transparent 35%
                ),
                linear-gradient(
                    125deg,
                    #102C48,
                    #14243B 65%,
                    #172C48
                );

            border: 1px solid rgba(65, 210, 230, 0.25);
            border-radius: 22px;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.25);
        }

        .hero-panel h3 {
            color: #FFFFFF !important;
            font-size: 27px;
            font-weight: 800;
        }

        .hero-panel p {
            color: #B7C9DC;
            line-height: 1.8;
            font-size: 15px;
            max-width: 750px;
        }

        /* INFORMATION PANELS */

        .info-panel {
            min-height: 175px;
            padding: 24px;
            margin: 10px 0;

            background: linear-gradient(
                145deg,
                #16263D,
                #101B2D
            );

            border: 1px solid rgba(100, 180, 210, 0.15);
            border-radius: 19px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.20);
            transition: all 0.25s ease;
        }

        .info-panel:hover {
            border-color: rgba(0, 210, 230, 0.45);
            transform: translateY(-3px);
            box-shadow: 0 15px 35px rgba(0, 170, 200, 0.08);
        }

        .info-panel h3 {
            font-size: 19px;
            margin-bottom: 12px;
            color: #49D7E8 !important;
        }

        .info-panel p {
            color: #AFC0D2;
            line-height: 1.7;
            font-size: 14px;
        }

        /* METRIC CARDS */

        [data-testid="stMetric"] {
            background: linear-gradient(
                145deg,
                #192D47,
                #122138
            );
            border: 1px solid rgba(70, 190, 220, 0.18);
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }

        [data-testid="stMetricLabel"] {
            color: #91A9C2 !important;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #52D9E9 !important;
            font-weight: 800;
        }

        /* GENERAL BUTTONS */

        div.stButton > button {
            min-height: 45px;
            border-radius: 12px;
            border: 1px solid rgba(70, 210, 230, 0.35);

            background: linear-gradient(
                110deg,
                #087EAB,
                #12B9C9
            );

            color: #FFFFFF;
            font-weight: 700;
            transition: all 0.2s ease;
            box-shadow: 0 7px 18px rgba(0, 180, 210, 0.15);
        }

        div.stButton > button:hover {
            background: linear-gradient(
                110deg,
                #079FC7,
                #36D5DD
            );
            color: #FFFFFF;
            border-color: #55E8F0;
            transform: translateY(-2px);
        }

        /* =================================================
           NEW CLICKABLE DASHBOARD CARDS
           ================================================= */

        .st-key-home_draw_card button,
        .st-key-home_upload_card button,
        .st-key-home_results_card button,
        .st-key-home_analytics_card button,
        .st-key-home_about_card button {

            width: 100%;
            min-height: 175px;

            display: flex;
            align-items: flex-start;
            justify-content: center;

            text-align: left;
            white-space: pre-wrap;

            padding: 23px;

            background: linear-gradient(
                145deg,
                #1B3150,
                #111E33
            );

            border: 1px solid rgba(80, 180, 220, 0.24);
            border-radius: 20px;

            color: #FFFFFF;
            font-size: 17px;
            font-weight: 800;

            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.20);

            transition:
                transform 0.25s ease,
                border-color 0.25s ease,
                box-shadow 0.25s ease,
                background 0.25s ease;
        }

        .st-key-home_draw_card button:hover,
        .st-key-home_upload_card button:hover,
        .st-key-home_results_card button:hover,
        .st-key-home_analytics_card button:hover,
        .st-key-home_about_card button:hover {

            background: linear-gradient(
                145deg,
                #21466A,
                #162B48
            );

            border: 1px solid #38C9DF;

            box-shadow:
                0 15px 35px rgba(0, 190, 220, 0.16);

            transform: translateY(-6px);
        }

        .st-key-home_draw_card button p,
        .st-key-home_upload_card button p,
        .st-key-home_results_card button p,
        .st-key-home_analytics_card button p,
        .st-key-home_about_card button p {
            line-height: 1.8;
        }

        /* CARD SUBTITLES */

        .dashboard-subtitle {
            color: #94ABC2;
            font-size: 13px;
            line-height: 1.6;
            margin-top: -7px;
            margin-bottom: 18px;
        }

        .dashboard-number {
            color: #42D9E4;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 2px;
        }

        /* FILE UPLOADER */

        [data-testid="stFileUploader"] section {
            background: #14243A;
            border: 1px dashed rgba(60, 200, 220, 0.45);
            border-radius: 18px;
            padding: 25px;
        }

        [data-testid="stFileUploader"] section:hover {
            border-color: #43D5E5;
        }

        /* INPUTS */

        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div,
        [data-baseweb="select"] > div {
            background: #14243A !important;
            border-color: rgba(100, 180, 210, 0.25) !important;
            border-radius: 10px !important;
        }

        /* PROGRESS BAR */

        [data-testid="stProgressBar"] > div > div {
            background: linear-gradient(
                90deg,
                #0789B7,
                #42D9E4
            );
        }

        /* DRAWING CANVAS */

        iframe[title="streamlit_drawable_canvas.st_canvas"] {
            max-width: 100%;
            border: 2px solid rgba(55, 205, 225, 0.35);
            border-radius: 17px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
        }

        /* DATA TABLE */

        [data-testid="stDataFrame"] {
            border: 1px solid rgba(70, 190, 220, 0.2);
            border-radius: 14px;
            overflow: hidden;
        }

        /* EXPANDER */

        [data-testid="stExpander"] {
            background: #14243A;
            border: 1px solid rgba(70, 190, 220, 0.18);
            border-radius: 14px;
        }

        .muted-copy {
            color: #91A9C2;
            line-height: 1.7;
        }

        /* FOOTER */

        .bottom-project-bar {
            position: fixed;
            z-index: 999;
            left: max(17rem, calc((100vw - 1450px)/2 + 2rem));
            right: max(2rem, calc((100vw - 1450px)/2 + 2rem));
            bottom: 0;

            padding: 15px 22px;

            display: flex;
            justify-content: space-between;
            gap: 15px;

            color: #A9BED2;
            font-size: 12px;

            background: rgba(13, 28, 47, 0.97);
            border: 1px solid rgba(70, 190, 220, 0.2);
            border-bottom: none;
            border-radius: 14px 14px 0 0;
            backdrop-filter: blur(15px);
        }

        @media (max-width: 900px) {

            [data-testid="stMainBlockContainer"] {
                margin: 8px;
                padding: 20px 15px 90px;
            }

            .bottom-project-bar {
                left: 8px;
                right: 8px;
            }

            .st-key-home_draw_card button,
            .st-key-home_upload_card button,
            .st-key-home_results_card button,
            .st-key-home_analytics_card button,
            .st-key-home_about_card button {
                min-height: 150px;
            }
        }

        @media (max-width: 600px) {

            .hero-panel {
                padding: 22px;
            }

            .hero-panel h3 {
                font-size: 21px;
            }

            .bottom-project-bar {
                font-size: 10px;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# HEADER
# ==========================================================

def render_header():

    st.markdown(
        '<div class="app-kicker">ARTIFICIAL INTELLIGENCE / DEEP LEARNING</div>',
        unsafe_allow_html=True,
    )

    st.title("HANDWRITTEN CHARACTER RECOGNITION")

    st.caption(
        "Intelligent Character Classification using Convolutional Neural Networks"
    )


# ==========================================================
# HOME DASHBOARD - REDESIGNED CLICKABLE CARDS
# ==========================================================

def navigate_to(page):

    st.session_state["navigation"] = page
    st.rerun()


def render_home():

    st.subheader("Overview Dashboard")

    # HERO SECTION

    st.markdown(
        """
        <div class="hero-panel">

            <h3>Welcome to Character AI</h3>

            <p>
                Experience intelligent handwritten character recognition
                using a trained Convolutional Neural Network.
                Draw your character or upload an image and let the AI
                identify the character automatically.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # PERFORMANCE METRICS

    metrics = load_json(RESULTS_DIR / "evaluation.json")
    history = load_json(RESULTS_DIR / "training_history.json")

    if metrics:

        columns = st.columns(3)

        columns[0].metric(
            "Test Accuracy",
            f"{metrics['test_accuracy']:.2%}"
        )

        if history:
            columns[1].metric(
                "Training Accuracy",
                f"{history['train_accuracy'][-1]:.2%}"
            )
        else:
            columns[1].metric("Training Accuracy", "N/A")

        columns[2].metric(
            "Test Samples",
            f"{metrics['test_sample_count']:,}"
        )

    else:

        st.info("Evaluation results are not available yet.")

    # NEW NAVIGATION AREA

    st.markdown("## Explore Your Workspace")

    st.markdown(
        '<p class="dashboard-subtitle">'
        'Choose an option below to start using the character recognition system.'
        '</p>',
        unsafe_allow_html=True,
    )

    # FIRST ROW

    col1, col2 = st.columns(2, gap="large")

    with col1:

        if st.button(
            "✍️   DRAW CHARACTER\n\n"
            "Create handwritten characters using the interactive drawing canvas.\n\n"
            "01  ·  CREATE",
            key="home_draw_card",
            use_container_width=True,
        ):

            navigate_to("Draw Character")

    with col2:

        if st.button(
            "🖼️   UPLOAD IMAGE\n\n"
            "Upload a handwritten image and let the CNN recognize it.\n\n"
            "02  ·  UPLOAD",
            key="home_upload_card",
            use_container_width=True,
        ):

            navigate_to("Upload Image")

    st.write("")

    # SECOND ROW

    col3, col4 = st.columns(2, gap="large")

    with col3:

        if st.button(
            "🔍   PREDICTION RESULTS\n\n"
            "Explore recognized characters and their confidence scores.\n\n"
            "03  ·  RESULTS",
            key="home_results_card",
            use_container_width=True,
        ):

            navigate_to("Prediction Results")

    with col4:

        if st.button(
            "📊   MODEL ANALYTICS\n\n"
            "Explore CNN accuracy, training graphs and evaluation results.\n\n"
            "04  ·  ANALYTICS",
            key="home_analytics_card",
            use_container_width=True,
        ):

            navigate_to("Model Performance")

    st.write("")

    # THIRD ROW

    if st.button(
        "🧠   ABOUT THE PROJECT\n\n"
        "Discover the CNN architecture, dataset and technologies used in this project.\n\n"
        "05  ·  EXPLORE",
        key="home_about_card",
        use_container_width=True,
    ):

        navigate_to("About Project")

    # DATASET INFORMATION

    st.markdown("---")

    st.markdown("### Supported Character Classes")

    st.caption("EMNIST Balanced Dataset — 47 classes")

    st.code(" ".join(EMNIST_BALANCED_LABELS))


# ==========================================================
# DRAW CHARACTER
# ==========================================================

def render_draw_page():

    st.subheader("Draw Your Character")

    st.markdown(
        '<p class="muted-copy">Draw one handwritten character below. '
        'Use the drawing tools to create or erase strokes.</p>',
        unsafe_allow_html=True,
    )

    if "drawing_mode" not in st.session_state:
        st.session_state.drawing_mode = "draw"

    if "canvas_generation" not in st.session_state:
        st.session_state.canvas_generation = 0

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Draw", key="draw_mode"):
            st.session_state.drawing_mode = "draw"

    with c2:
        if st.button("Eraser", key="eraser_mode"):
            st.session_state.drawing_mode = "eraser"

    with c3:
        if st.button("Clear Canvas", key="clear_canvas"):
            st.session_state.canvas_generation += 1
            st.session_state.drawing_mode = "draw"
            st.rerun()

    is_eraser = st.session_state.drawing_mode == "eraser"

    stroke_width = st.slider(
        "Stroke size",
        4,
        28,
        14,
    )

    stroke_color = "#10121B" if is_eraser else "#FFFFFF"

    canvas_col, guide_col = st.columns([1, 1])

    with canvas_col:

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#10121B",
            height=320,
            width=320,
            drawing_mode="freedraw",
            update_streamlit=True,
            display_toolbar=False,
            key=f"canvas_{st.session_state.canvas_generation}",
        )

    with guide_col:

        st.markdown(
            """
            <div class="info-panel">
                <h3>Drawing Instructions</h3>
                <p>
                    1. Draw only one character.<br><br>
                    2. Keep the character centered.<br><br>
                    3. Use a clear and visible stroke.<br><br>
                    4. Remove unwanted marks using the eraser.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button(
        "Predict Character",
        type="primary",
        key="predict_canvas",
    ):

        if canvas_result.image_data is None:
            st.warning("Please draw a character first.")
        else:
            run_prediction(
                canvas_result.image_data,
                canvas_result.image_data,
                "drawing",
            )


# ==========================================================
# UPLOAD IMAGE
# ==========================================================

def render_upload_page():

    st.subheader("Upload Handwritten Image")

    st.markdown(
        '<p class="muted-copy">Upload a PNG or JPG image containing '
        'one handwritten character.</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        key="character_upload",
    )

    if uploaded_file is None:
        return

    image_bytes = uploaded_file.getvalue()

    try:
        image = load_image(image_bytes)
    except ValueError as error:
        st.error(str(error))
        return

    st.image(
        image,
        caption="Uploaded Character",
        width=320,
    )

    if st.button(
        "Predict Character",
        type="primary",
        key="predict_upload",
    ):

        run_prediction(
            image_bytes,
            image,
            "uploaded image",
        )


# ==========================================================
# PREDICTION RESULTS
# ==========================================================

def render_prediction_page():

    st.subheader("AI Prediction Results")

    result = st.session_state.get("last_prediction")

    if result is None:

        st.info(
            "Draw or upload a character to view prediction results."
        )
        return

    top_prediction = result.top_predictions[0]

    original_image = st.session_state.get(
        "last_input_image"
    )

    image_col, result_col = st.columns([1, 1.3])

    with image_col:

        if original_image is not None:
            st.image(
                original_image,
                caption="Original Image",
                width=300,
            )

        st.image(
            result.processed_image,
            caption="Processed Input",
            width=196,
        )

    with result_col:

        st.markdown(
            """
            <div class="hero-panel">
                <h3>Recognition Complete</h3>
                <p>The CNN model has analyzed your handwritten character.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.metric(
            "Predicted Character",
            top_prediction.character,
            f"{top_prediction.confidence:.2%} confidence",
        )

        st.markdown("### Top Predictions")

        for index, prediction in enumerate(
            result.top_predictions,
            start=1,
        ):

            st.write(
                f"**{index}. {prediction.character}**"
            )

            st.progress(prediction.confidence)

            st.caption(
                f"Confidence: {prediction.confidence:.2%}"
            )


# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

def render_performance_page():

    st.subheader("Model Performance Analytics")

    metrics = load_json(
        RESULTS_DIR / "evaluation.json"
    )

    history = load_json(
        RESULTS_DIR / "training_history.json"
    )

    if metrics is None:
        st.warning("Evaluation metrics are not available.")
        return

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Test Accuracy",
        f"{metrics['test_accuracy']:.2%}"
    )

    c2.metric(
        "Test Samples",
        f"{metrics['test_sample_count']:,}"
    )

    if history:
        c3.metric(
            "Best Validation Accuracy",
            f"{history['best_validation_accuracy']:.2%}"
        )

    accuracy_path = RESULTS_DIR / "accuracy.png"
    loss_path = RESULTS_DIR / "loss.png"

    left, right = st.columns(2)

    if accuracy_path.is_file():
        left.image(
            str(accuracy_path),
            caption="Accuracy Graph",
        )

    if loss_path.is_file():
        right.image(
            str(loss_path),
            caption="Loss Graph",
        )

    confusion_path = RESULTS_DIR / "confusion_matrix.png"

    if confusion_path.is_file():
        st.image(
            str(confusion_path),
            caption="Confusion Matrix",
        )

    report = metrics.get("classification_report", {})

    rows = [
        {"Class": name, **values}
        for name, values in report.items()
        if isinstance(values, dict)
    ]

    if rows:
        with st.expander("Detailed Classification Report"):
            st.dataframe(
                rows,
                width="stretch",
                hide_index=True,
            )


# ==========================================================
# ABOUT PROJECT
# ==========================================================

def render_about_page():

    st.subheader("About the Project")

    st.markdown(
        """
        <div class="hero-panel">
            <h3>Handwritten Character Recognition Using CNN</h3>
            <p>
                A deep learning application designed to identify handwritten
                digits and letters using image classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="info-panel">
                <h3>Technology Stack</h3>
                <p>
                    Python<br>
                    PyTorch<br>
                    Torchvision<br>
                    OpenCV<br>
                    NumPy<br>
                    Streamlit<br>
                    Matplotlib<br>
                    Scikit-learn
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="info-panel">
                <h3>Deep Learning Model</h3>
                <p>
                    Convolutional Neural Network<br>
                    ReLU Activation<br>
                    Max Pooling<br>
                    Fully Connected Layers<br>
                    Dropout<br>
                    Softmax-based class probabilities
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Dataset")

    st.write(
        "EMNIST Balanced — 47 handwritten character classes."
    )

    st.markdown("### Project Workflow")

    st.write(
        "Image Input → Preprocessing → CNN Feature Extraction "
        "→ Classification → Prediction Results"
    )


# ==========================================================
# MAIN APPLICATION
# ==========================================================

def main():

    st.set_page_config(
        page_title="Character AI",
        page_icon="C",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_styles()

    if st.session_state.pop(
        "navigate_to_results",
        False,
    ):
        st.session_state["navigation"] = "Prediction Results"

    with st.sidebar:

        st.markdown("## CHARACTER AI")

        st.caption("DEEP LEARNING PLATFORM")

        st.divider()

        selected_page = st.radio(
            "Navigation",
            PAGES,
            label_visibility="collapsed",
            key="navigation",
        )

        st.divider()

        st.caption("MODEL INFORMATION")

        st.markdown("**Dataset:** EMNIST Balanced")

        st.markdown("**Classes:** 47")

        st.markdown("**Architecture:** CNN")

    render_header()

    page_renderers = {
        "Home Dashboard": render_home,
        "Draw Character": render_draw_page,
        "Upload Image": render_upload_page,
        "Prediction Results": render_prediction_page,
        "Model Performance": render_performance_page,
        "About Project": render_about_page,
    }

    page_renderers[selected_page]()

    st.markdown(
        """
        <div class="bottom-project-bar">
            <span>● &nbsp; CHARACTER AI &nbsp; | &nbsp; CNN PLATFORM</span>
            <span>EMNIST BALANCED &nbsp; | &nbsp; 47 CLASSES</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
