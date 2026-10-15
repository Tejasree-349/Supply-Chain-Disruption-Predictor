from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_PATH,
    MODEL_DIR,
    REPORT_DIR,
    PLOT_DIR,
    TARGET_COLUMN,
)

DATA_PATH = RAW_DATA_PATH
MODEL_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0b1020;
        color: #e8edf7;
    }

    [data-testid="stSidebar"] {
        background: #11182b;
        border-right: 1px solid #26324b;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.92);
    }

    .hero {
        padding: 30px;
        border-radius: 22px;
        background:
            linear-gradient(
                120deg,
                rgba(24, 42, 77, 0.98),
                rgba(26, 33, 60, 0.96)
            );
        border: 1px solid #344566;
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #75e0c3;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-size: 11px;
        font-weight: 800;
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(28px, 4vw, 42px);
        font-weight: 800;
        margin: 10px 0;
    }

    .hero p {
        color: #bdc9df;
        font-size: 14px;
        line-height: 1.8;
    }

    .metric-card {
        background: #141d32;
        border: 1px solid #2a3855;
        border-radius: 16px;
        padding: 20px;
        min-height: 130px;
    }

    .metric-label {
        color: #9baac4;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 29px;
        font-weight: 800;
        margin: 10px 0 5px 0;
    }

    .metric-note {
        color: #75e0c3;
        font-size: 12px;
    }

    .section-title {
        color: #ffffff;
        font-size: 23px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #a8b5cb;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .insight-card {
        background: #141d32;
        border: 1px solid #2a3855;
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 10px;
    }

    .insight-card h4 {
        color: #75e0c3;
        font-size: 14px;
    }

    .insight-card p {
        color: #d3dcee;
        font-size: 13px;
        line-height: 1.7;
    }

    div[data-testid="stMetric"] {
        background: #141d32;
        border: 1px solid #2a3855;
        padding: 16px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a8b5cb;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }

    .stButton > button {
        background: #75e0c3;
        color: #0b1020;
        border: none;
        border-radius: 10px;
        font-weight: 800;
        min-height: 44px;
    }

    .stButton > button:hover {
        background: #a2f1dd;
        color: #0b1020;
    }

    .footer {
        color: #8190aa;
        font-size: 11px;
        text-align: center;
        padding: 30px 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PLOTLY THEME
# ============================================================

CHART_TEMPLATE = "plotly_dark"

CHART_COLORS = [
    "#75e0c3",
    "#8da8ff",
    "#ffc777",
    "#ff8295",
    "#b8a0ff",
    "#64c7e8",
]


def style_chart(fig, height=380):
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color="#dce5f5",
        ),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    fig.update_xaxes(
        gridcolor="#26324b",
        zerolinecolor="#26324b",
    )

    fig.update_yaxes(
        gridcolor="#26324b",
        zerolinecolor="#26324b",
    )

    return fig


def section_heading(title, subtitle):
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner="Loading supply chain data...")
def load_dataset(path, modified_time):
    if not Path(path).exists():
        return None

    data = pd.read_csv(
        path,
        encoding="latin-1",
        low_memory=False,
    )

    data.columns = (
        data.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    return data


df = load_dataset(
    str(DATA_PATH),
    DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0,
)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_saved_models(model_dir):
    loaded = {}

    for path in Path(model_dir).glob("*.joblib"):
        try:
            loaded[path.stem] = joblib.load(path)
        except Exception:
            continue

    return loaded


models = load_saved_models(str(MODEL_DIR))

# ============================================================
# LOAD EXISTING EVALUATION REPORT
# ============================================================

@st.cache_data
def load_evaluation_report(path, modified_time):
    if not Path(path).exists():
        return pd.DataFrame()

    return pd.read_csv(path)


evaluation_path = REPORT_DIR / "model_evaluation.csv"

evaluation_df = load_evaluation_report(
    str(evaluation_path),
    evaluation_path.stat().st_mtime
    if evaluation_path.exists()
    else 0,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:12px 0 22px 0;">
            <div style="font-size:11px;color:#75e0c3;
                        letter-spacing:3px;font-weight:800;">
            </div>
            <div style="font-size:22px;font-weight:800;
                        color:#ffffff;margin-top:5px;">
                Supply Chain
            </div>
            <div style="font-size:22px;font-weight:800;
                        color:#8da8ff;">
                Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    pages = [
        "Executive Overview",
        "Dataset Intelligence",
        "Model Performance",
        "Prediction Audit",
        "AI Insights",
        "Prediction Lab",
    ]

    page = st.radio(
        "NAVIGATION",
        pages,
        index=0,
    )

    st.markdown("---")

    st.markdown("### System Status")

    st.write(
        "🟢 Dataset available"
        if df is not None
        else "🔴 Dataset unavailable"
    )

    st.write(
        f"🟢 Models loaded: {len(models)}"
        if models
        else "🟠 No trained models"
    )

    st.caption("Supply Chain Risk Analytics • ML Dashboard")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            AI-POWERED OPERATIONS ANALYTICS
        </div>
        <h1>Supply Chain Risk Intelligence</h1>
        <p>
            Monitor historical delivery patterns, inspect machine-learning
            performance, understand prediction errors, and estimate the
            delivery risk of new orders.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if df is None:
    st.error(
        "Dataset not found. Place DataCoSupplyChainDataset.csv "
        "inside data/raw/."
    )
    st.stop()

if TARGET_COLUMN not in df.columns:
    st.error(
        f"Target column '{TARGET_COLUMN}' was not found. "
        "Check the dataset column names."
    )
    st.stop()

# Normalize target for analytics
target_values = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce",
)

valid_target_mask = target_values.isin([0, 1])

analytics_df = df.loc[valid_target_mask].copy()
analytics_df[TARGET_COLUMN] = target_values[valid_target_mask].astype(int)

if analytics_df.empty:
    st.error("No valid binary target values were found.")
    st.stop()

# ============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    section_heading(
        "Executive Overview",
        "A high-level view of observed data and available model results.",
    )

    total_orders = len(analytics_df)

    late_count = int(
        (analytics_df[TARGET_COLUMN] == 1).sum()
    )

    observed_rate = (
        analytics_df[TARGET_COLUMN].mean() * 100
    )

    f1_value = None
    auc_value = None

    if not evaluation_df.empty:
        if "f1_score" in evaluation_df.columns:
            f1_value = evaluation_df["f1_score"].max()

        if "roc_auc" in evaluation_df.columns:
            auc_value = evaluation_df["roc_auc"].max()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Orders analyzed",
            f"{total_orders:,}",
            "Records with valid target labels",
        )

    with c2:
        metric_card(
            "Observed risk rate",
            f"{observed_rate:.1f}%",
            f"{late_count:,} positive labels",
        )

    with c3:
        metric_card(
            "Best recorded F1",
            f"{f1_value:.3f}" if f1_value is not None else "N/A",
            "From saved evaluation report",
        )

    with c4:
        metric_card(
            "Best recorded ROC-AUC",
            f"{auc_value:.3f}" if auc_value is not None else "N/A",
            "From saved evaluation report",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        section_heading(
            "Observed Target Distribution",
            "Actual target labels in the loaded dataset.",
        )

        target_chart = (
            analytics_df[TARGET_COLUMN]
            .value_counts()
            .reindex([0, 1], fill_value=0)
            .rename_axis("Risk")
            .reset_index(name="Orders")
        )

        target_chart["Risk"] = target_chart["Risk"].map({
            0: "No late-delivery risk",
            1: "Late-delivery risk",
        })

        fig = px.pie(
            target_chart,
            names="Risk",
            values="Orders",
            hole=0.65,
            color_discrete_sequence=CHART_COLORS,
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        st.plotly_chart(
            style_chart(fig, 360),
            use_container_width=True,
        )

    with right:
        section_heading(
            "Model Comparison",
            "Metrics from the saved evaluation report.",
        )

        if not evaluation_df.empty:
            metric_options = [
                c for c in [
                    "f1_score",
                    "precision",
                    "recall",
                    "roc_auc",
                    "average_precision",
                ]
                if c in evaluation_df.columns
            ]

            chosen_metric = st.selectbox(
                "Metric",
                metric_options,
                key="overview_metric",
            )

            fig = px.bar(
                evaluation_df,
                x="model",
                y=chosen_metric,
                color="model",
                color_discrete_sequence=CHART_COLORS,
                text_auto=".3f",
            )

            st.plotly_chart(
                style_chart(fig, 360),
                use_container_width=True,
            )
        else:
            st.info(
                "No evaluation report found. Train and evaluate "
                "the models to display performance metrics."
            )

    section_heading(
        "What the dataset tells us",
        "Descriptive findings computed directly from the loaded data.",
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(
            """
            <div class="insight-card">
                <h4>Dataset coverage</h4>
                <p>
                    The dashboard summarizes the records currently
                    loaded from the configured CSV file.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>Positive labels</h4>
                <p>
                    {late_count:,} records have a target label of 1.
                    This is an observed label count, not a new model
                    prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_c:
        st.markdown(
            """
            <div class="insight-card">
                <h4>Model monitoring</h4>
                <p>
                    Use the Model Performance and Prediction Audit
                    sections to inspect recorded metrics and errors.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE 2: DATASET INTELLIGENCE
# ============================================================

elif page == "Dataset Intelligence":

    section_heading(
        "Dataset Intelligence",
        "Explore the observed distribution of orders and late-delivery risk.",
    )

    left, right = st.columns(2)

    with left:
        category_columns = [
            column for column in [
                "Shipping_Mode",
                "Category_Name",
                "Order_Region",
                "Market",
                "Order_Country",
            ]
            if column in analytics_df.columns
        ]

        if category_columns:
            selected_column = st.selectbox(
                "Analyze category",
                category_columns,
            )

            grouped = (
                analytics_df.groupby(
                    selected_column,
                    dropna=False,
                )[TARGET_COLUMN]
                .agg(["count", "mean"])
                .reset_index()
            )

            grouped.columns = [
                selected_column,
                "Order_Count",
                "Risk_Rate",
            ]

            grouped["Risk_Rate"] *= 100

            grouped = grouped.sort_values(
                "Risk_Rate",
                ascending=False,
            ).head(20)

            fig = px.bar(
                grouped,
                x="Risk_Rate",
                y=selected_column,
                orientation="h",
                color="Risk_Rate",
                color_continuous_scale=[
                    "#8da8ff",
                    "#75e0c3",
                    "#ff8295",
                ],
                hover_data=["Order_Count"],
                title=f"Observed risk by {selected_column}",
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                style_chart(fig, 550),
                use_container_width=True,
            )
        else:
            st.info(
                "No recognized categorical columns were found."
            )

    with right:
        numeric_columns = analytics_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        numeric_columns = [
            column for column in numeric_columns
            if column != TARGET_COLUMN
        ]

        if numeric_columns:
            selected_numeric = st.selectbox(
                "Numerical feature",
                numeric_columns,
            )

            fig = px.histogram(
                analytics_df,
                x=selected_numeric,
                color=TARGET_COLUMN,
                nbins=40,
                barmode="overlay",
                opacity=0.75,
                color_discrete_sequence=CHART_COLORS,
                title=f"Distribution of {selected_numeric}",
            )

            st.plotly_chart(
                style_chart(fig, 550),
                use_container_width=True,
            )

    st.markdown("### Filter and inspect the dataset")

    filter_column = st.selectbox(
        "Optional categorical filter",
        ["All records"] + [
            c for c in [
                "Shipping_Mode",
                "Category_Name",
                "Order_Region",
                "Market",
            ]
            if c in analytics_df.columns
        ],
    )

    filtered_df = analytics_df.copy()

    if filter_column != "All records":
        values = sorted(
            filtered_df[filter_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        chosen_values = st.multiselect(
            f"Select {filter_column} values",
            values,
            default=values[: min(5, len(values))],
        )

        filtered_df = filtered_df[
            filtered_df[filter_column]
            .astype(str)
            .isin(chosen_values)
        ]

    st.write(f"Records displayed: {len(filtered_df):,}")

    st.dataframe(
        filtered_df.head(500),
        use_container_width=True,
        height=350,
    )

    st.download_button(
        "Download filtered records",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_supply_chain_data.csv",
        mime="text/csv",
    )


# ============================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    section_heading(
        "Model Performance",
        "Inspect saved evaluation results and, when possible, "
        "calculate performance from labeled test records.",
    )

    if not evaluation_df.empty:
        st.markdown("### Saved model comparison")

        st.dataframe(
            evaluation_df,
            use_container_width=True,
        )

        available_metrics = [
            column for column in [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "average_precision",
            ]
            if column in evaluation_df.columns
        ]

        metric_choice = st.selectbox(
            "Compare metric",
            available_metrics,
            key="performance_metric",
        )

        fig = px.bar(
            evaluation_df,
            x="model",
            y=metric_choice,
            color="model",
            text_auto=".4f",
            color_discrete_sequence=CHART_COLORS,
        )

        st.plotly_chart(
            style_chart(fig, 420),
            use_container_width=True,
        )

    else:
        st.warning(
            "The saved model_evaluation.csv report is missing. "
            "Run model training and evaluation first."
        )

    test_path = REPORT_DIR / "test_data.csv"

    st.markdown("### Recalculate test performance")

    if test_path.exists() and models:
        test_df = pd.read_csv(test_path)

        if TARGET_COLUMN in test_df.columns:
            y_true = pd.to_numeric(
                test_df[TARGET_COLUMN],
                errors="coerce",
            )

            valid = y_true.isin([0, 1])

            y_true = y_true.loc[valid].astype(int)
            X_test = test_df.loc[valid].drop(
                columns=[TARGET_COLUMN]
            )

            model_choice = st.selectbox(
                "Choose a trained model",
                list(models.keys()),
                key="recalc_model",
            )

            if st.button("Calculate test metrics"):
                model = models[model_choice]

                try:
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "Accuracy",
                        f"{accuracy_score(y_true, y_pred):.3f}",
                    )

                    c2.metric(
                        "Precision",
                        f"{precision_score(y_true, y_pred, zero_division=0):.3f}",
                    )

                    c3.metric(
                        "Recall",
                        f"{recall_score(y_true, y_pred, zero_division=0):.3f}",
                    )

                    c4.metric(
                        "F1-score",
                        f"{f1_score(y_true, y_pred, zero_division=0):.3f}",
                    )

                    cm = confusion_matrix(
                        y_true,
                        y_pred,
                        labels=[0, 1],
                    )

                    fig = go.Figure(
                        data=go.Heatmap(
                            z=cm,
                            x=["Predicted 0", "Predicted 1"],
                            y=["Actual 0", "Actual 1"],
                            text=cm,
                            texttemplate="%{text}",
                            colorscale="Teal",
                        )
                    )

                    fig.update_layout(
                        title="Confusion Matrix",
                        xaxis_title="Prediction",
                        yaxis_title="Actual label",
                    )

                    st.plotly_chart(
                        style_chart(fig, 420),
                        use_container_width=True,
                    )

                    if len(np.unique(y_true)) == 2:
                        fpr, tpr, _ = roc_curve(
                            y_true,
                            y_prob,
                        )

                        fig = go.Figure()

                        fig.add_trace(
                            go.Scatter(
                                x=fpr,
                                y=tpr,
                                mode="lines",
                                name="Model ROC",
                                line=dict(
                                    color="#75e0c3",
                                    width=3,
                                ),
                            )
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=[0, 1],
                                y=[0, 1],
                                mode="lines",
                                name="Random baseline",
                                line=dict(
                                    dash="dash",
                                    color="#8da8ff",
                                ),
                            )
                        )

                        fig.update_layout(
                            title=(
                                "ROC Curve | AUC = "
                                f"{roc_auc_score(y_true, y_prob):.3f}"
                            ),
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                        )

                        st.plotly_chart(
                            style_chart(fig, 420),
                            use_container_width=True,
                        )

                except Exception as error:
                    st.error(
                        "Could not calculate metrics. "
                        "Check that the saved model and test records "
                        f"are compatible. Details: {error}"
                    )

    else:
        st.info(
            "Test data or trained models are unavailable. "
            "Run the training and evaluation pipeline first."
        )


# ============================================================
# PAGE 4: PREDICTION AUDIT
# ============================================================

elif page == "Prediction Audit":

    section_heading(
        "Prediction Audit",
        "Inspect where a model succeeds and where it misses risk.",
    )

    test_path = REPORT_DIR / "test_data.csv"

    if not test_path.exists() or not models:
        st.warning(
            "Train and evaluate the models first to create the "
            "test dataset and saved model artifacts."
        )

    else:
        test_df = pd.read_csv(test_path)

        if TARGET_COLUMN not in test_df.columns:
            st.error("Test data has no target column.")
        else:
            model_choice = st.selectbox(
                "Select model for audit",
                list(models.keys()),
                key="audit_model",
            )

            model = models[model_choice]

            y_true = pd.to_numeric(
                test_df[TARGET_COLUMN],
                errors="coerce",
            )

            valid = y_true.isin([0, 1])

            audit_df = test_df.loc[valid].copy()
            y_true = y_true.loc[valid].astype(int)

            X_audit = audit_df.drop(
                columns=[TARGET_COLUMN]
            )

            try:
                y_pred = model.predict(X_audit)
                y_prob = model.predict_proba(X_audit)[:, 1]

                audit_df["Actual_Label"] = y_true.to_numpy()
                audit_df["Predicted_Label"] = y_pred
                audit_df["Risk_Probability"] = y_prob

                audit_df["Prediction_Result"] = np.select(
                    [
                        (y_true.to_numpy() == 1) & (y_pred == 1),
                        (y_true.to_numpy() == 0) & (y_pred == 0),
                        (y_true.to_numpy() == 0) & (y_pred == 1),
                        (y_true.to_numpy() == 1) & (y_pred == 0),
                    ],
                    [
                        "True Positive",
                        "True Negative",
                        "False Positive",
                        "False Negative",
                    ],
                    default="Unknown",
                )

                counts = (
                    audit_df["Prediction_Result"]
                    .value_counts()
                    .to_dict()
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Correct predictions",
                    f"{int((audit_df['Actual_Label'] == audit_df['Predicted_Label']).sum()):,}",
                )

                c2.metric(
                    "Incorrect predictions",
                    f"{int((audit_df['Actual_Label'] != audit_df['Predicted_Label']).sum()):,}",
                )

                c3.metric(
                    "Missed high-risk orders",
                    f"{counts.get('False Negative', 0):,}",
                )

                c4.metric(
                    "False alarms",
                    f"{counts.get('False Positive', 0):,}",
                )

                fig = px.histogram(
                    audit_df,
                    x="Risk_Probability",
                    color="Prediction_Result",
                    nbins=35,
                    barmode="overlay",
                    opacity=0.75,
                    color_discrete_sequence=CHART_COLORS,
                    title="Predicted probability by audit outcome",
                )

                st.plotly_chart(
                    style_chart(fig, 420),
                    use_container_width=True,
                )

                outcome_filter = st.selectbox(
                    "Show prediction outcome",
                    [
                        "All outcomes",
                        "True Positive",
                        "True Negative",
                        "False Positive",
                        "False Negative",
                    ],
                )

                display_df = audit_df.copy()

                if outcome_filter != "All outcomes":
                    display_df = display_df[
                        display_df["Prediction_Result"]
                        == outcome_filter
                    ]

                st.dataframe(
                    display_df.head(1000),
                    use_container_width=True,
                    height=400,
                )

                st.download_button(
                    "Download prediction audit",
                    data=display_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="prediction_audit.csv",
                    mime="text/csv",
                )

            except Exception as error:
                st.error(
                    "Audit failed. Check the model and test data "
                    f"compatibility. Details: {error}"
                )


# ============================================================
# PAGE 5: AI INSIGHTS
# ============================================================

elif page == "AI Insights":

    section_heading(
        "AI Insights",
        "Understand which input features influence model performance.",
    )

    importance_path = (
        REPORT_DIR / "permutation_importance.csv"
    )

    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)

        if {
            "Feature",
            "Importance_Mean",
        }.issubset(importance_df.columns):

            importance_df = importance_df.sort_values(
                "Importance_Mean",
                ascending=False,
            )

            top_n = st.slider(
                "Number of features",
                min_value=5,
                max_value=min(
                    30,
                    max(5, len(importance_df)),
                ),
                value=min(15, max(5, len(importance_df))),
            )

            top_df = importance_df.head(top_n).sort_values(
                "Importance_Mean",
                ascending=True,
            )

            fig = px.bar(
                top_df,
                x="Importance_Mean",
                y="Feature",
                orientation="h",
                error_x="Importance_Std"
                if "Importance_Std" in top_df.columns
                else None,
                color="Importance_Mean",
                color_continuous_scale=[
                    "#8da8ff",
                    "#75e0c3",
                ],
                title="Permutation Feature Importance",
            )

            st.plotly_chart(
                style_chart(fig, 600),
                use_container_width=True,
            )

            st.dataframe(
                importance_df.head(top_n),
                use_container_width=True,
            )

        else:
            st.warning(
                "The importance report has an unexpected format."
            )

    else:
        st.info(
            "Feature importance report not found. Run "
            "python -m src.explainability first."
        )

    st.markdown("### Interpretation guide")

    st.markdown(
        """
        <div class="insight-card">
            <h4>Positive importance</h4>
            <p>
                Shuffling a feature reduced the model's evaluation
                score. The model may rely on that feature.
            </p>
        </div>

        <div class="insight-card">
            <h4>Near-zero importance</h4>
            <p>
                Shuffling the feature had little measured effect
                on the selected evaluation metric.
            </p>
        </div>

        <div class="insight-card">
            <h4>Negative importance</h4>
            <p>
                The score improved after shuffling the feature.
                The feature may be noisy, redundant, or unstable.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Importance is not causation. Correlated features can "
        "share or obscure predictive information."
    )


# ============================================================
# PAGE 6: PREDICTION LAB
# ============================================================

elif page == "Prediction Lab":

    section_heading(
        "Prediction Lab",
        "Score new hypothetical orders or upload a CSV for batch prediction.",
    )

    if not models:
        st.warning(
            "No trained models are available. Run "
            "python -m src.train first."
        )
        st.stop()

    model_choice = st.selectbox(
        "Choose prediction model",
        list(models.keys()),
        key="prediction_model",
    )

    selected_model = models[model_choice]

    st.markdown("### Option A — Upload a CSV")

    uploaded_file = st.file_uploader(
        "Upload orders to score",
        type=["csv"],
        help=(
            "Use the feature columns expected by the selected model. "
            "Do not include the target column."
        ),
    )

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)

            st.write("Uploaded data preview")
            st.dataframe(
                upload_df.head(20),
                use_container_width=True,
            )

            if st.button("Predict uploaded orders"):
                probabilities = selected_model.predict_proba(
                    upload_df
                )[:, 1]

                predictions = selected_model.predict(
                    upload_df
                )

                result_df = upload_df.copy()

                result_df["Predicted_Late_Delivery_Risk"] = predictions
                result_df["Risk_Probability"] = probabilities

                result_df["Risk_Level"] = np.select(
                    [
                        probabilities >= 0.75,
                        probabilities >= 0.40,
                    ],
                    [
                        "HIGH",
                        "MEDIUM",
                    ],
                    default="LOW",
                )

                st.success(
                    f"Generated predictions for {len(result_df):,} records."
                )

                st.dataframe(
                    result_df,
                    use_container_width=True,
                )

                st.download_button(
                    "Download prediction results",
                    data=result_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="new_order_predictions.csv",
                    mime="text/csv",
                )

        except Exception as error:
            st.error(
                "The uploaded data could not be scored. "
                "Ensure its columns and data types match the trained "
                f"model. Details: {error}"
            )

    st.divider()

    st.markdown("### Option B — Enter a hypothetical order")

    st.info(
        "The form below is generated from the selected model's "
        "actual training schema. Predictions will only be possible "
        "if the model can accept the entered feature values."
    )

    # Recover feature schema from the fitted ColumnTransformer.
    try:
        preprocessor = selected_model.named_steps["preprocessor"]

        feature_names = list(preprocessor.feature_names_in_)

    except Exception:
        feature_names = []

    if not feature_names:
        st.warning(
            "The selected model does not expose a recoverable input "
            "schema. Use the CSV option with correctly formatted columns."
        )

    else:
        # Determine numerical/categorical fields using the fitted
        # preprocessing pipeline.
        numeric_features = []
        categorical_features = []

        for name, transformer, columns in preprocessor.transformers_:
            if name == "numeric":
                numeric_features.extend(list(columns))
            elif name == "categorical":
                categorical_features.extend(list(columns))

        # Retrieve sensible default values from historical data.
        historical_features = analytics_df.reindex(
            columns=feature_names
        )

        with st.form("manual_prediction_form"):

            st.write("Enter available order information")

            manual_values = {}

            left, right = st.columns(2)

            for index, feature in enumerate(feature_names):
                container = left if index % 2 == 0 else right

                with container:
                    if feature in numeric_features:
                        source = pd.to_numeric(
                            historical_features[feature],
                            errors="coerce",
                        ).dropna()

                        default_value = (
                            float(source.median())
                            if not source.empty
                            else 0.0
                        )

                        if not np.isfinite(default_value):
                            default_value = 0.0

                        manual_values[feature] = st.number_input(
                            feature.replace("_", " "),
                            value=default_value,
                            key=f"manual_{feature}",
                        )

                    elif feature in categorical_features:
                        source = (
                            historical_features[feature]
                            .dropna()
                            .astype(str)
                        )

                        choices = sorted(
                            source.unique().tolist()
                        )

                        if choices:
                            manual_values[feature] = st.selectbox(
                                feature.replace("_", " "),
                                choices,
                                key=f"manual_{feature}",
                            )
                        else:
                            manual_values[feature] = st.text_input(
                                feature.replace("_", " "),
                                key=f"manual_{feature}",
                            )

                    else:
                        manual_values[feature] = st.text_input(
                            feature.replace("_", " "),
                            key=f"manual_{feature}",
                        )

            submitted = st.form_submit_button(
                "Analyze order risk"
            )

        if submitted:
            try:
                input_df = pd.DataFrame(
                    [manual_values],
                    columns=feature_names,
                )

                probabilities = selected_model.predict_proba(
                    input_df
                )[:, 1]

                predictions = selected_model.predict(
                    input_df
                )

                probability = float(probabilities[0])
                prediction = int(predictions[0])

                if probability >= 0.75:
                    risk_level = "HIGH"
                    recommendation = (
                        "Prioritize this order for manual review. "
                        "Check carrier capacity, inventory readiness, "
                        "and available alternatives."
                    )

                elif probability >= 0.40:
                    risk_level = "MEDIUM"
                    recommendation = (
                        "Monitor fulfillment progress and review "
                        "possible inventory or carrier bottlenecks."
                    )

                else:
                    risk_level = "LOW"
                    recommendation = (
                        "Continue standard monitoring and reassess "
                        "if new risk signals appear."
                    )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Predicted label",
                    str(prediction),
                )

                c2.metric(
                    "Estimated risk probability",
                    f"{probability:.1%}",
                )

                c3.metric(
                    "Risk classification",
                    risk_level,
                )

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        number={"suffix": "%"},
                        title={
                            "text": "Estimated late-delivery risk"
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100],
                            },
                            "bar": {
                                "color": "#75e0c3",
                            },
                            "steps": [
                                {
                                    "range": [0, 40],
                                    "color": "#1d3b3a",
                                },
                                {
                                    "range": [40, 75],
                                    "color": "#59472a",
                                },
                                {
                                    "range": [75, 100],
                                    "color": "#592c42",
                                },
                            ],
                        },
                    )
                )

                st.plotly_chart(
                    style_chart(fig, 330),
                    use_container_width=True,
                )

                st.markdown("### Recommended action")

                st.markdown(
                    f"""
                    <div class="insight-card">
                        <h4>{risk_level} RISK</h4>
                        <p>{recommendation}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    "This is a model estimate for a hypothetical order. "
                    "It is not a confirmed delivery outcome or guarantee."
                )

            except Exception as error:
                st.error(
                    "Could not generate the prediction. Check the "
                    "entered values and model input schema. "
                    f"Details: {error}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Supply Chain Intelligence |
        Machine-learning estimates are not guaranteed outcomes.
    </div>
    """,
    unsafe_allow_html=True,
)