"""Iris Flower Classification - Streamlit application.

Predictions come from a Random Forest model that was trained beforehand and
saved as `random_forest_model.pkl`. This app only loads that file; it never
retrains or overwrites it.

Run with:  streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon=":herb:",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "random_forest_model.pkl"

# Canonical feature order of sklearn's load_iris(). The saved model is assumed
# to expect the same order unless it stores its own feature names (see
# `make_model_input`, which checks this).
FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
FEATURE_KEYS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SHORT_NAMES = ["Sepal length", "Sepal width", "Petal length", "Petal width"]

SPECIES = ["Iris setosa", "Iris versicolor", "Iris virginica"]
# Restrained botanical palette (dark green, sage, muted ochre)
COLORS = {"Iris setosa": "#2F5D50", "Iris versicolor": "#8FA98A", "Iris virginica": "#B08D57"}
BG = "#F7F6F1"
INK = "#2B2F2E"

# Minimal CSS: serif headings and a quieter sidebar. Nothing structural.
st.markdown(
    """
    <style>
    h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; color: #2F4F3E; font-weight: 600; }
    h1 { font-size: 2.1rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #D5DBCB; }
    .small-note { color: #5C6560; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data and model helpers (kept separate from the page code)
# --------------------------------------------------------------------------
@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Original Iris dataset from scikit-learn, with a readable species column."""
    iris = load_iris(as_frame=True)
    df = iris.data.copy()
    df.columns = FEATURES
    df["species"] = pd.Categorical(
        [SPECIES[i] for i in iris.target], categories=SPECIES, ordered=True
    )
    return df


@st.cache_resource
def load_model():
    """Load the saved model once per session. Returns (model, error_message)."""
    if not MODEL_PATH.exists():
        return None, (
            f"Model file not found: `{MODEL_PATH.name}`. Place it in the same folder as "
            f"`app.py` ({MODEL_PATH.parent})."
        )
    try:
        import pickle

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except ModuleNotFoundError as e:
        return None, f"The model needs a package that is not installed: `{e.name}`."
    except Exception:  # pickle can fail in several ways (version mismatch, bad file...)
        try:
            import joblib

            model = joblib.load(MODEL_PATH)
        except Exception as e2:
            return None, (
                "The model file could not be loaded. It may have been saved with a different "
                f"scikit-learn version. Details: {type(e2).__name__}: {e2}"
            )
    if not hasattr(model, "predict"):
        return None, "The loaded object has no `predict` method, so it is not a trained model."
    n_in = getattr(model, "n_features_in_", 4)
    if n_in != 4:
        return None, f"The model expects {n_in} input features, but the Iris data has 4."
    return model, None


def species_name(label) -> str:
    """Convert a model class label (0/1/2 or a string) to 'Iris <name>'."""
    if isinstance(label, (int, np.integer)):
        return SPECIES[int(label)]
    text = str(label).lower()
    for s in SPECIES:
        if s.split()[1] in text:
            return s
    return str(label)


def _feature_key(name: str):
    """Map a column name such as 'sepal length (cm)' or 'SepalLengthCm' to a canonical key."""
    n = name.lower().replace("_", "").replace(" ", "")
    part = "sepal" if "sepal" in n else "petal" if "petal" in n else None
    dim = "length" if "length" in n else "width" if "width" in n else None
    return f"{part}_{dim}" if part and dim else None


def make_model_input(model, X: pd.DataFrame):
    """Arrange X (columns in FEATURES order) the way the saved model expects.

    - If the model stores feature names, columns are matched by name when all four
      can be identified; otherwise they are passed by position.
    - If it stores none (trained on a NumPy array), a plain array is passed.
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return X.to_numpy()
    names = [str(n) for n in names]
    keys = [_feature_key(n) for n in names]
    if sorted(k for k in keys if k) == sorted(FEATURE_KEYS):
        by_key = dict(zip(FEATURE_KEYS, X.columns))
        return pd.DataFrame({n: X[by_key[k]].to_numpy() for n, k in zip(names, keys)})
    out = X.copy()
    out.columns = names  # positional fallback
    return out


def alignment_note(model) -> str:
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return "The model stores no feature names; inputs are passed by position in the order shown above."
    keys = [_feature_key(str(n)) for n in names]
    if sorted(k for k in keys if k) == sorted(FEATURE_KEYS):
        return f"Feature names stored in the model were matched to the inputs: {', '.join(map(str, names))}."
    return (
        "The model's feature names could not be matched automatically "
        f"({', '.join(map(str, names))}); inputs are passed by position."
    )


def to_model_labels(model, species: pd.Series) -> np.ndarray:
    """Dataset species -> the label type the model uses (needed for cross-validation)."""
    classes = list(model.classes_)
    if all(isinstance(c, (int, np.integer)) for c in classes):
        return species.map({s: i for i, s in enumerate(SPECIES)}).to_numpy()
    lookup = {species_name(c): c for c in classes}
    return species.map(lookup).to_numpy()


@st.cache_data
def cross_validate(_model, n_splits: int = 5):
    """5-fold stratified CV using an *unfitted clone* of the saved model's configuration.

    The saved model itself is not touched. This estimates how well this model
    configuration generalises to unseen flowers.
    """
    df = load_dataset()
    X = df[FEATURES]
    y = to_model_labels(_model, df["species"])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_val_score(clone(_model), make_model_input(_model, X), y, cv=cv)



def show_table(data, **kwargs):
    """st.dataframe at full width, working on both older and current Streamlit versions."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:  # older Streamlit without width="stretch"
        st.dataframe(data, use_container_width=True, **kwargs)


# --------------------------------------------------------------------------
# Chart helpers
# --------------------------------------------------------------------------
def new_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(BG)
    for a in np.atleast_1d(ax).ravel():
        a.set_facecolor(BG)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color("#9AA39C")
        a.tick_params(colors=INK, labelsize=9)
        a.xaxis.label.set_color(INK)
        a.yaxis.label.set_color(INK)
        a.title.set_color(INK)
    return fig, ax


# --------------------------------------------------------------------------
# Page 1: Project Overview
# --------------------------------------------------------------------------
def page_overview():
    st.title("Iris Flower Classification")
    st.markdown(
        "This application predicts the species of an Iris flower from four measurements "
        "of its sepals and petals. The prediction is made by a Random Forest classifier "
        "that was trained beforehand and saved to a file, which the app loads and queries."
    )

    img = Path(__file__).parent / "assets" / "iris.jpg"  # optional: add your own photo
    if img.exists():
        st.image(str(img), width=420)

    st.divider()
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("The Random Forest algorithm")
        st.markdown(
            "A Random Forest is an ensemble of many decision trees. Each tree is trained on a "
            "random resample of the training data and considers only a random subset of the "
            "features at each split. To classify a new sample, every tree votes, and the forest "
            "returns the most common class. Averaging the trees' outputs also gives an estimated "
            "probability for each class. Combining many varied trees reduces the overfitting "
            "that a single deep tree tends to show."
        )
    with right:
        st.subheader("Species")
        st.markdown("\n".join(f"- *{s}*" for s in SPECIES))
        st.subheader("Input features")
        st.markdown("\n".join(f"- {f.capitalize()}" for f in FEATURES))

    st.divider()
    st.subheader("Workflow")
    c1, a1, c2, a2, c3 = st.columns([4, 1, 4, 1, 4])
    with c1:
        with st.container(border=True):
            st.markdown("**1. Input**")
            st.caption("Four measurements in centimetres")
    with a1:
        st.markdown("<h3 style='text-align:center;margin-top:0.6rem'>&rarr;</h3>", unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.markdown("**2. Trained model**")
            st.caption("Saved Random Forest loaded from file")
    with a2:
        st.markdown("<h3 style='text-align:center;margin-top:0.6rem'>&rarr;</h3>", unsafe_allow_html=True)
    with c3:
        with st.container(border=True):
            st.markdown("**3. Output**")
            st.caption("Predicted species and class probabilities")

    st.divider()
    st.subheader("Purpose")
    st.markdown(
        "The Iris dataset is a standard introductory problem in machine learning. This project "
        "uses it to show the full path from exploring data to serving predictions from a saved "
        "model. It is a teaching example, not a botanical identification tool: it only knows "
        "these three species and the four measurements above."
    )
    st.caption("Use the sidebar to open the dataset summary or to test a new sample.")


# --------------------------------------------------------------------------
# Page 2: Dataset Summary
# --------------------------------------------------------------------------
def page_dataset(model):
    df = load_dataset()
    st.title("Dataset Summary")
    st.markdown("Exploration of the original Iris dataset (Fisher, 1936) as shipped with scikit-learn.")

    # ---- A. Overview ------------------------------------------------------
    st.header("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(df))
    c2.metric("Features", len(FEATURES))
    c3.metric("Classes", len(SPECIES))

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Samples per species**")
        counts = df["species"].value_counts().reindex(SPECIES)
        show_table(counts.rename("Samples").to_frame())
    with right:
        st.markdown("**Features and units**")
        show_table(
            pd.DataFrame({"Feature": SHORT_NAMES, "Unit": ["cm"] * 4}).set_index("Feature")
        )

    # ---- B. Statistics ----------------------------------------------------
    st.header("Statistical summary")
    stats = df[FEATURES].describe().T
    stats.columns = ["Count", "Mean", "Std", "Min", "25%", "Median", "75%", "Max"]
    stats["Count"] = stats["Count"].astype(int)
    stats.index = SHORT_NAMES
    show_table(stats.round(2))
    st.caption("All values in centimetres. Std is the sample standard deviation.")

    # ---- C. Visualisations -----------------------------------------------
    st.header("Visualizations")

    v1, v2 = st.columns(2, gap="large")
    with v1:
        fig, ax = new_fig(figsize=(5, 3.6))
        counts = df["species"].value_counts().reindex(SPECIES)
        ax.bar([s.replace("Iris ", "") for s in SPECIES], counts.values, color=[COLORS[s] for s in SPECIES], width=0.55)
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.8, str(v), ha="center", color=INK, fontsize=9)
        ax.set_ylim(0, counts.max() * 1.15)
        ax.set_xlabel("Species")
        ax.set_ylabel("Number of samples")
        ax.set_title("Samples per species", loc="left", fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with v2:
        fig, ax = new_fig(figsize=(5, 3.6))
        for s in SPECIES:
            sub = df[df["species"] == s]
            ax.scatter(sub["petal length (cm)"], sub["petal width (cm)"], s=28, alpha=0.85,
                       color=COLORS[s], label=s, edgecolor=BG, linewidth=0.5)
        ax.set_xlabel("Petal length (cm)")
        ax.set_ylabel("Petal width (cm)")
        ax.set_title("Petal length vs. petal width", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Box plots: one panel per feature, species side by side
    fig, axes = new_fig(1, 4, figsize=(11, 3.6))
    for ax, feat, short in zip(axes, FEATURES, SHORT_NAMES):
        data = [df.loc[df["species"] == s, feat] for s in SPECIES]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops={"color": INK, "linewidth": 1.4})
        for patch, s in zip(bp["boxes"], SPECIES):
            patch.set_facecolor(COLORS[s])
            patch.set_alpha(0.9)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([s.replace("Iris ", "") for s in SPECIES], rotation=20)
        ax.set_title(short, fontsize=10)
        ax.set_ylabel("cm" if ax is axes[0] else "")
    fig.suptitle("Feature distributions by species", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Correlation heatmap
    corr = df[FEATURES].corr()
    fig, ax = new_fig(figsize=(5.2, 4.2))
    im = ax.imshow(corr.values, cmap="Greens", vmin=-1, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(SHORT_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(SHORT_NAMES)
    for i in range(4):
        for j in range(4):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if val > 0.6 else INK)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("Correlation between features (Pearson)", loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    h1, _ = st.columns([1, 1])
    with h1:
        st.pyplot(fig)
    plt.close(fig)

    # ---- D. Preview -------------------------------------------------------
    st.header("Dataset preview")
    f1, f2 = st.columns([3, 1])
    chosen = f1.multiselect("Filter by species", SPECIES, default=SPECIES)
    n_rows = f2.number_input("Rows to show", min_value=5, max_value=150, value=10, step=5)
    view = df[df["species"].isin(chosen)].head(int(n_rows))
    show_table(view.reset_index(drop=True), height=300)
    st.caption(f"Showing {len(view)} of {int(df['species'].isin(chosen).sum())} matching rows.")

    # ---- E. Model information --------------------------------------------
    st.header("Model information")
    if model is None:
        st.warning("The saved model is not loaded, so model information is unavailable. See the sidebar for details.")
        return

    st.markdown(
        f"Predictions in this app are generated by the saved model (`{MODEL_PATH.name}`, "
        f"type `{type(model).__name__}`). It is loaded as-is and never retrained."
    )
    st.caption(alignment_note(model))
    st.caption(f"Supports class probabilities: {'yes' if hasattr(model, 'predict_proba') else 'no'}.")

    st.markdown("**Evaluation**")
    try:
        scores = cross_validate(model)
        # Agreement with dataset labels. The training split of the saved model is not
        # known, so this number may include samples it was trained on.
        pred = [species_name(p) for p in model.predict(make_model_input(model, df[FEATURES]))]
        agree = float(np.mean(np.array(pred) == df["species"].astype(str).to_numpy()))

        e1, e2 = st.columns(2)
        e1.metric("5-fold cross-validated accuracy", f"{scores.mean() * 100:.1f}%",
                  help=f"Per fold: {', '.join(f'{s * 100:.1f}%' for s in scores)}")
        e2.metric("Agreement on the full dataset", f"{agree * 100:.1f}%")
        st.markdown(
            "<p class='small-note'><b>Cross-validated accuracy</b> is computed here: the data is split into "
            "5 stratified folds, and a fresh copy of the model's configuration is fitted on 4 folds and "
            "scored on the held-out fold, repeated for each fold. It estimates performance on unseen "
            "flowers. The saved model itself is not refitted.<br>"
            "<b>Agreement on the full dataset</b> compares the saved model's predictions with all 150 labels. "
            "Because it is not known which samples the model was trained on, this is not a test "
            "accuracy and is likely optimistic.</p>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Evaluation could not be computed: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# Page 3: Test a New Sample
# --------------------------------------------------------------------------
def page_predict(model):
    df = load_dataset()
    st.title("Test a New Sample")
    st.markdown("Enter the four measurements of a flower and run the saved model on them.")

    if model is None:
        st.error("The model could not be loaded, so predictions are unavailable. See the sidebar for details.")
        return

    # Slider ranges: observed min/max with a small margin; default = dataset mean.
    spec = {}
    for feat in FEATURES:
        lo, hi = float(df[feat].min()), float(df[feat].max())
        spec[feat] = (round(max(0.1, lo - 0.5), 1), round(hi + 1.0, 1), round(float(df[feat].mean()), 1))

    # Widget state lives in session_state so the Reset button can restore defaults.
    for i, feat in enumerate(FEATURES):
        st.session_state.setdefault(f"in_{i}", spec[feat][2])

    def reset():
        for i, feat in enumerate(FEATURES):
            st.session_state[f"in_{i}"] = spec[feat][2]
        st.session_state.pop("result", None)

    col_in, col_out = st.columns([2, 3], gap="large")

    with col_in:
        st.subheader("Measurements")
        values = []
        for i, feat in enumerate(FEATURES):
            lo, hi, _ = spec[feat]
            values.append(st.slider(feat.capitalize(), min_value=lo, max_value=hi, step=0.1, key=f"in_{i}"))
        b1, b2 = st.columns(2)
        run = b1.button("Predict Species", type="primary")
        b2.button("Reset", on_click=reset)

        # Input sanity check against the observed data range
        for feat, v in zip(FEATURES, values):
            lo, hi = float(df[feat].min()), float(df[feat].max())
            if not lo <= v <= hi:
                st.caption(f"{feat.capitalize()} = {v} is outside the range seen in the dataset ({lo}-{hi}).")

    if run:
        try:
            X = pd.DataFrame([values], columns=FEATURES)
            model_in = make_model_input(model, X)
            pred = species_name(model.predict(model_in)[0])
            proba = None
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(model_in)[0]
                proba = {species_name(c): float(v) for c, v in zip(model.classes_, p)}
            st.session_state["result"] = {"values": list(values), "pred": pred, "proba": proba}
        except Exception as e:
            st.session_state.pop("result", None)
            col_out.error(
                f"Prediction failed ({type(e).__name__}: {e}). Check that the model expects "
                "four numeric features in the order sepal length, sepal width, petal length, petal width."
            )

    with col_out:
        result = st.session_state.get("result")
        # Hide a stale result once the sliders no longer match what was predicted
        if not result or result["values"] != list(values):
            st.subheader("Prediction")
            st.caption("Set the measurements and press Predict Species.")
            return

        pred = result["pred"]
        st.subheader("Prediction")
        st.markdown(
            f"<div style='border-left:5px solid {COLORS.get(pred, '#2F5D50')};padding:0.4rem 1rem;'>"
            f"<span class='small-note'>Predicted species</span><br>"
            f"<span style='font-family:Georgia,serif;font-size:2rem;color:#2F4F3E'><i>{pred}</i></span></div>",
            unsafe_allow_html=True,
        )

        if result["proba"]:
            st.markdown("**Class probabilities**")
            proba = result["proba"]
            order = [s for s in SPECIES if s in proba] or list(proba)
            fig, ax = new_fig(figsize=(5.5, 2.2))
            vals = [proba[s] * 100 for s in order]
            ax.barh([s for s in order][::-1], vals[::-1], color=[COLORS.get(s, "#8FA98A") for s in order][::-1], height=0.55)
            for y, v in enumerate(vals[::-1]):
                ax.text(min(v + 1.5, 88), y, f"{v:.1f}%", va="center", fontsize=9, color=INK)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "These are the model's estimated probabilities (the average of the trees' class "
                "estimates), not a guarantee about the real flower."
            )
        else:
            st.caption("This model does not provide class probabilities.")

        # Factual comparison with the dataset; this is not an explanation of the forest's internals.
        st.markdown("**How the entered values compare with the dataset**")
        ref = df[df["species"] == pred]
        rows = []
        for feat, short, v in zip(FEATURES, SHORT_NAMES, result["values"]):
            lo, hi = ref[feat].min(), ref[feat].max()
            rows.append({
                "Feature": short,
                "Entered (cm)": v,
                f"{pred} mean (cm)": round(ref[feat].mean(), 2),
                f"{pred} range (cm)": f"{lo:.1f} - {hi:.1f}",
                "Within range": "yes" if lo <= v <= hi else "no",
            })
        show_table(pd.DataFrame(rows).set_index("Feature"))
        st.caption("Descriptive comparison with the 50 dataset samples of the predicted species. "
                   "It does not show how the Random Forest reached its decision.")


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
model, model_error = load_model()

with st.sidebar:
    st.markdown("### Iris Classification")
    st.caption("Machine learning project")
    page = st.radio(
        "Navigation",
        ["Project Overview", "Dataset Summary", "Test a New Sample"],
        label_visibility="collapsed",
    )
    st.divider()
    if model_error:
        st.error(model_error)
    else:
        st.caption(f"Model: `{MODEL_PATH.name}` loaded ({type(model).__name__})")

if page == "Project Overview":
    page_overview()
elif page == "Dataset Summary":
    page_dataset(model)
else:
    page_predict(model)
