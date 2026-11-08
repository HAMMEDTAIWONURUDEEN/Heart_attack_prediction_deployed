
import streamlit as st
import pandas as pd
import joblib


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. APPLICATION DESIGN
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #07110F;
    color: #E8F5EF;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: #0B1915;
    border-right: 1px solid #1D3B30;
}

section[data-testid="stSidebar"] * {
    color: #E8F5EF;
}

h1, h2, h3 {
    color: #F0FFF8 !important;
    font-weight: 700 !important;
}

.stNumberInput label,
.stSelectbox label {
    color: #C5DED0 !important;
}

.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: #10251C !important;
    color: #F0FFF8 !important;
    border-color: #315D48 !important;
}

.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(90deg, #17B879, #26D99A) !important;
    color: #04140D !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 45px;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #42E6A9 !important;
}

div[data-testid="stMetric"] {
    background: #10221B;
    border: 1px solid #234737;
    border-radius: 15px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #91B3A2;
}

div[data-testid="stMetricValue"] {
    color: #39D99B;
}

hr {
    border-color: #234737 !important;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD THE TRAINED MODEL AND SCALER
# =====================================================

@st.cache_resource
def load_model():

    model = joblib.load("heart_attack_model.pkl")

    scaler = joblib.load("heart_attack_scaler.pkl")

    return model, scaler


try:

    model, scaler = load_model()

except Exception as error:

    st.error("The model or scaler could not be loaded.")

    st.exception(error)

    st.stop()


# =====================================================
# 4. REUSABLE DESIGN FUNCTIONS
# =====================================================

def render_html(content):

    st.html(content)


def page_header(label, title, description):

    render_html(f"""
    <div style="
        background: linear-gradient(120deg, #102B21, #0B1915);
        border: 1px solid #24523E;
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 25px;
    ">

        <div style="
            color: #39D99B;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 2px;
        ">
            {label}
        </div>

        <div style="
            font-size: 34px;
            font-weight: 800;
            color: #F0FFF8;
            margin-top: 12px;
            margin-bottom: 10px;
        ">
            {title}
        </div>

        <div style="
            color: #A7C7B7;
            font-size: 14px;
            line-height: 1.8;
            max-width: 750px;
        ">
            {description}
        </div>

    </div>
    """)


def section_heading(title, description):

    render_html(f"""
    <div style="margin-top: 28px; margin-bottom: 18px;">

        <div style="
            color: #F0FFF8;
            font-size: 21px;
            font-weight: 700;
        ">
            {title}
        </div>

        <div style="
            color: #91B3A2;
            font-size: 12px;
            margin-top: 7px;
        ">
            {description}
        </div>

    </div>
    """)


def info_card(title, description, icon):

    render_html(f"""
    <div style="
        background: #10221B;
        border: 1px solid #234737;
        border-radius: 16px;
        padding: 23px;
        min-height: 150px;
    ">

        <div style="font-size: 25px; margin-bottom: 15px;">
            {icon}
        </div>

        <div style="
            color: #F0FFF8;
            font-size: 17px;
            font-weight: 700;
            margin-bottom: 10px;
        ">
            {title}
        </div>

        <div style="
            color: #91B3A2;
            font-size: 12px;
            line-height: 1.8;
        ">
            {description}
        </div>

    </div>
    """)


def disclaimer():

    render_html("""
    <div style="
        background: #211F14;
        border: 1px solid #685B28;
        border-radius: 12px;
        padding: 16px;
        color: #DCCB8D;
        font-size: 12px;
        line-height: 1.8;
        margin-top: 22px;
    ">
        <strong>Important medical notice:</strong>
        This application has not been clinically validated.
        Its model output is not a medical diagnosis and must not
        be used to confirm or rule out a heart attack or make
        treatment decisions. Seek emergency medical care if
        heart attack symptoms are suspected, regardless of the
        model output.
    </div>
    """)


# =====================================================
# 5. SIDEBAR
# =====================================================

with st.sidebar:

    render_html("""
    <div style="padding: 15px 5px 25px 5px;">

        <div style="
            font-size: 25px;
            font-weight: 800;
            color: #F0FFF8;
        ">
            💚 CardioPredict
            <span style="color: #39D99B;">AI</span>
        </div>

        <div style="
            color: #7FA891;
            font-size: 10px;
            margin-top: 7px;
            letter-spacing: 1px;
        ">
            MACHINE LEARNING HEALTH ANALYTICS
        </div>

    </div>
    """)

    st.markdown("---")

    st.markdown("##### NAVIGATION")

    page = st.radio(
        "Choose a section",
        [
            "Overview",
            "Prediction Dashboard",
            "About the Model"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    render_html("""
    <div style="
        background: #10251C;
        padding: 17px;
        border-radius: 12px;
    ">

        <div style="
            color: #39D99B;
            font-size: 11px;
            font-weight: 700;
        ">
            SYSTEM STATUS
        </div>

        <div style="
            color: #E8F5EF;
            font-size: 13px;
            margin-top: 12px;
        ">
            ● Model loaded
        </div>

        <div style="
            color: #91B3A2;
            font-size: 11px;
            margin-top: 7px;
        ">
            Random Forest Classifier
        </div>

    </div>
    """)

    st.markdown("---")

    st.caption("CARDIOPREDICT AI · HEALTH ANALYTICS")


# =====================================================
# 6. OVERVIEW PAGE
# =====================================================

if page == "Overview":

    page_header(
        "CARDIOVASCULAR HEALTH ANALYTICS",
        "Smarter Health Data Analysis.",
        "Explore a machine-learning demonstration that analyses "
        "eight cardiovascular-related input features and produces "
        "a classification based on its training dataset."
    )

    section_heading(
        "System overview",
        "An overview of the technology powering this application."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            label="MACHINE LEARNING MODEL",
            value="Random Forest",
            delta="Classification"
        )

    with col2:

        st.metric(
            label="INPUT FEATURES",
            value="08",
            delta="Model variables"
        )

    with col3:

        st.metric(
            label="PREPROCESSING",
            value="StandardScaler",
            delta="Saved scaler"
        )

    section_heading(
        "Explore the system",
        "Discover the main components of CardioPredict AI."
    )

    left, right = st.columns(2)

    with left:

        info_card(
            "Cardiovascular Assessment",
            "Enter the eight model features and obtain a "
            "classification from the trained machine-learning model.",
            "♥"
        )

        st.write("")

        if st.button(
            "Open Prediction Dashboard",
            use_container_width=True
        ):

            st.session_state["page"] = "Prediction Dashboard"

            st.rerun()

    with right:

        info_card(
            "Model Information",
            "Explore the Random Forest algorithm, input features "
            "and preprocessing pipeline.",
            "◈"
        )

        st.write("")

        if st.button(
            "Explore Model Information",
            use_container_width=True
        ):

            st.session_state["page"] = "About the Model"

            st.rerun()

    disclaimer()


# =====================================================
# 7. PREDICTION DASHBOARD
# =====================================================

elif page == "Prediction Dashboard":

    page_header(
        "MODEL PREDICTION CENTRE",
        "Cardiovascular Assessment.",
        "Provide the input values below to run a prediction "
        "using the trained Random Forest model."
    )

    section_heading(
        "Patient input parameters",
        "Enter values using the same units and format as the original dataset."
    )

    with st.form("prediction_form"):

        render_html("""
        <div style="
            color: #39D99B;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 15px;
        ">
            01 · PERSONAL INFORMATION
        </div>
        """)

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30,
                step=1
            )

        with col2:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

        st.markdown("---")

        render_html("""
        <div style="
            color: #39D99B;
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 15px;
        ">
            02 · CARDIOVASCULAR INDICATORS
        </div>
        """)

        col3, col4 = st.columns(2)

        with col3:

            heart_rate = st.number_input(
                "Heart Rate",
                min_value=0.0,
                value=75.0,
                step=1.0
            )

            systolic_bp = st.number_input(
                "Systolic Blood Pressure",
                min_value=0.0,
                value=120.0,
                step=1.0
            )

            diastolic_bp = st.number_input(
                "Diastolic Blood Pressure",
                min_value=0.0,
                value=80.0,
                step=1.0
            )

        with col4:

            blood_sugar = st.number_input(
                "Blood Sugar",
                min_value=0.0,
                value=100.0,
                step=1.0
            )

            ck_mb = st.number_input(
                "CK-MB",
                min_value=0.0,
                value=2.0,
                step=0.1
            )

            troponin = st.number_input(
                "Troponin",
                min_value=0.0,
                value=0.01,
                step=0.01,
                format="%.4f"
            )

        st.markdown("---")

        submitted = st.form_submit_button(
            "Run Model Prediction",
            use_container_width=True
        )

    if submitted:

        gender_value = 1 if gender == "Male" else 0

        input_data = pd.DataFrame(
            [[
                age,
                gender_value,
                heart_rate,
                systolic_bp,
                diastolic_bp,
                blood_sugar,
                ck_mb,
                troponin
            ]],
            columns=[
                "Age",
                "Gender",
                "Heart rate",
                "Systolic blood pressure",
                "Diastolic blood pressure",
                "Blood sugar",
                "CK-MB",
                "Troponin"
            ]
        )

        try:

            # Transform input using the saved scaler

            scaled_input = scaler.transform(input_data)

            # Generate model prediction

            prediction = int(
                model.predict(scaled_input)[0]
            )

            # Prepare classification information

            if prediction == 1:

                result_title = "Positive Classification"

                result_description = (
                    "The model classified the submitted input "
                    "as positive according to the target labels "
                    "in its training dataset."
                )

                result_color = "#FF7777"
                result_background = "#251A1A"
                result_border = "#A84A4A"
                result_status = "POSITIVE"

            else:

                result_title = "Negative Classification"

                result_description = (
                    "The model classified the submitted input "
                    "as negative according to the target labels "
                    "in its training dataset."
                )

                result_color = "#39D99B"
                result_background = "#10291F"
                result_border = "#27875B"
                result_status = "NEGATIVE"

            # Display assessment results

            section_heading(
                "Assessment Results",
                "Classification generated by the trained machine-learning model."
            )

            render_html(f"""
            <div style="
                background: {result_background};
                border: 1px solid {result_border};
                border-radius: 18px;
                padding: 30px;
                margin-top: 10px;
            ">

                <div style="
                    color: {result_color};
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 2px;
                ">
                    MODEL CLASSIFICATION
                </div>

                <div style="
                    color: #F0FFF8;
                    font-size: 30px;
                    font-weight: 800;
                    margin-top: 15px;
                ">
                    {result_title}
                </div>

                <div style="
                    color: #B7CEC1;
                    font-size: 14px;
                    line-height: 1.9;
                    margin-top: 15px;
                    max-width: 800px;
                ">
                    {result_description}
                </div>

                <div style="
                    display: inline-block;
                    background: {result_background};
                    border: 1px solid {result_border};
                    border-radius: 7px;
                    padding: 7px 13px;
                    margin-top: 20px;
                    color: {result_color};
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 1px;
                ">
                    {result_status}
                </div>

            </div>
            """)

            # Display assessment summary

            st.markdown("")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    label="INPUT FEATURES",
                    value="08"
                )

            with col2:

                st.metric(
                    label="MODEL",
                    value="Random Forest"
                )

            with col3:

                st.metric(
                    label="OUTPUT CLASS",
                    value=result_status
                )

            # Display submitted input values

            with st.expander("View submitted input values"):

                display_data = input_data.copy()

                display_data["Gender"] = gender

                st.dataframe(
                    display_data,
                    use_container_width=True,
                    hide_index=True
                )

            # Generate downloadable assessment report

            report = f"""
CARDIOPREDICT AI
CARDIOVASCULAR ASSESSMENT REPORT

========================================

ASSESSMENT RESULT

Model Classification: {result_title}

Output Class: {result_status}

========================================

SUBMITTED INPUT VALUES

Age: {age}
Gender: {gender}
Heart Rate: {heart_rate}
Systolic Blood Pressure: {systolic_bp}
Diastolic Blood Pressure: {diastolic_bp}
Blood Sugar: {blood_sugar}
CK-MB: {ck_mb}
Troponin: {troponin}

========================================

MODEL INFORMATION

Algorithm: Random Forest Classifier
Input Features: 8
Preprocessing: StandardScaler

========================================

IMPORTANT MEDICAL NOTICE

This application has not been clinically validated.

The model classification is not a medical diagnosis.
It cannot confirm or rule out a heart attack and must
not be used to make treatment decisions.

If heart attack symptoms are suspected, seek emergency
medical care immediately, regardless of this model output.

========================================

CardioPredict AI
Built with Python, Streamlit and Scikit-learn.
"""

            st.markdown("")

            st.download_button(
                label="Download Assessment Report",
                data=report,
                file_name="CardioPredict_Assessment_Report.txt",
                mime="text/plain",
                use_container_width=True
            )

            # Medical safety notice

            render_html("""
            <div style="
                background: #211F14;
                border: 1px solid #685B28;
                border-radius: 12px;
                padding: 18px;
                color: #DCCB8D;
                font-size: 12px;
                line-height: 1.9;
                margin-top: 22px;
            ">

                <strong>Important Medical Notice</strong>

                <br><br>

                This application has not been clinically validated.
                Its model output is not a medical diagnosis and
                cannot confirm or rule out a heart attack.

                <br><br>

                A negative classification does not guarantee that
                someone is free from a heart condition. If heart
                attack symptoms are suspected, seek emergency
                medical care regardless of the model output.

            </div>
            """)

        except Exception as error:

            st.error("Prediction failed.")

            st.exception(error)


# =====================================================
# 8. ABOUT THE MODEL PAGE
# =====================================================

elif page == "About the Model":

    page_header(
        "TECHNICAL INFORMATION",
        "Inside the Machine Learning Model.",
        "Understand the algorithm, features and preprocessing "
        "steps used by this application."
    )

    section_heading(
        "Model architecture",
        "The machine-learning technology behind the predictions."
    )

    info_card(
        "Random Forest Classifier",
        "A supervised machine-learning algorithm that combines "
        "multiple decision trees to produce a classification.",
        "🧠"
    )

    section_heading(
        "Model configuration",
        "The model configuration used in the training project."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Algorithm",
            "Random Forest"
        )

        st.metric(
            "Maximum Tree Depth",
            "10"
        )

    with col2:

        st.metric(
            "Minimum Samples Split",
            "5"
        )

        st.metric(
            "Random State",
            "42"
        )

    section_heading(
        "Input features",
        "The eight variables expected by the trained model."
    )

    features = [
        "Age",
        "Gender",
        "Heart rate",
        "Systolic blood pressure",
        "Diastolic blood pressure",
        "Blood sugar",
        "CK-MB",
        "Troponin"
    ]

    for index, feature in enumerate(features, start=1):

        render_html(f"""
        <div style="
            background: #10221B;
            border: 1px solid #234737;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 9px;
        ">

            <span style="
                color: #39D99B;
                font-weight: 700;
            ">
                {index:02d}
            </span>

            <span style="
                color: #E8F5EF;
                margin-left: 15px;
                font-size: 13px;
            ">
                {feature}
            </span>

        </div>
        """)

    section_heading(
        "Prediction pipeline",
        "How the application processes an input."
    )

    pipeline_steps = [
        ("01", "Input collection", "The user enters eight feature values."),
        ("02", "Data preparation", "The application creates a DataFrame in the correct feature order."),
        ("03", "Feature scaling", "The saved StandardScaler transforms the input."),
        ("04", "Model inference", "The trained Random Forest processes the transformed input."),
        ("05", "Classification", "The model returns its predicted class.")
    ]

    for number, title, description in pipeline_steps:

        render_html(f"""
        <div style="
            background: #10221B;
            border: 1px solid #234737;
            border-radius: 12px;
            padding: 17px;
            margin-bottom: 10px;
        ">

            <div style="
                color: #39D99B;
                font-size: 11px;
                font-weight: 700;
            ">
                STEP {number}
            </div>

            <div style="
                color: #F0FFF8;
                font-size: 16px;
                font-weight: 700;
                margin-top: 7px;
            ">
                {title}
            </div>

            <div style="
                color: #91B3A2;
                font-size: 12px;
                margin-top: 6px;
            ">
                {description}
            </div>

        </div>
        """)

    disclaimer()


# =====================================================
# 9. FOOTER
# =====================================================

st.markdown("---")

render_html("""
<div style="
    text-align: center;
    color: #668777;
    font-size: 11px;
    padding: 15px;
    line-height: 2;
">

    <strong>CARDIOPREDICT AI</strong>

    <br>

    Built with Python, Streamlit and Scikit-learn.

    <br>

    Cardiovascular Data Analysis · Not a medical diagnostic system.

</div>
""")
