import streamlit as st

from src.predict import predict_student


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(0, 180, 255, 0.08), transparent 28%),
        radial-gradient(circle at 85% 20%, rgba(130, 80, 255, 0.08), transparent 30%),
        #080b12;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    text-align: center;
    padding: 35px 20px;
    margin-bottom: 25px;
    border: 1px solid rgba(0, 200, 255, 0.22);
    border-radius: 20px;
    background: rgba(10, 15, 25, 0.80);
    box-shadow: 0 0 40px rgba(0, 180, 255, 0.08);
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: 2px;
    background: linear-gradient(90deg, #00d9ff, #7c5cff, #00d9ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #aeb8c8;
    font-size: 1.05rem;
    letter-spacing: 1px;
    margin-top: 10px;
}

.section-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #e8edf7;
    margin-top: 25px;
    margin-bottom: 15px;
}

.result-card {
    text-align: center;
    padding: 35px;
    margin-top: 25px;
    border-radius: 20px;
    border: 1px solid rgba(0, 220, 255, 0.28);
    background: linear-gradient(
        135deg,
        rgba(0, 180, 255, 0.10),
        rgba(120, 80, 255, 0.10)
    );
    box-shadow: 0 0 45px rgba(0, 180, 255, 0.10);
}

.result-label {
    color: #9ba8ba;
    font-size: 1rem;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.result-score {
    font-size: 4.5rem;
    font-weight: 800;
    color: #00d9ff;
    margin: 10px 0;
}

.result-category {
    font-size: 1.2rem;
    color: #dce4f2;
}

.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 12px;
    border: 1px solid rgba(0, 220, 255, 0.5);
    background: linear-gradient(90deg, #008dcc, #6548d8);
    color: white;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 1px;
}

.stButton > button:hover {
    box-shadow: 0 8px 25px rgba(0, 180, 255, 0.20);
}

[data-testid="stMetric"] {
    background: rgba(15, 20, 32, 0.75);
    border: 1px solid rgba(120, 140, 170, 0.15);
    padding: 15px;
    border-radius: 14px;
}

.footer {
    text-align: center;
    color: #667085;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(120, 140, 170, 0.12);
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
<div class="hero-title">🎓 STUDENT PERFORMANCE AI</div>
<div class="hero-subtitle">MACHINE LEARNING • REGRESSION • ACADEMIC PERFORMANCE PREDICTION</div>
</div>
""",
    unsafe_allow_html=True,
)

st.info(
    "Enter the student's academic, demographic, family, and lifestyle "
    "information to estimate the final G3 grade."
)


# ============================================================
# ACADEMIC PROFILE
# ============================================================

st.markdown(
    '<div class="section-title">📚 Academic & Student Profile</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("#### Academic")

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=22,
        value=17,
    )

    studytime = st.selectbox(
        "Study Time",
        [1, 2, 3, 4],
        index=1,
    )

    failures = st.selectbox(
        "Previous Failures",
        [0, 1, 2, 3],
        index=0,
    )

    absences = st.number_input(
        "Absences",
        min_value=0,
        max_value=100,
        value=5,
    )


with col2:

    st.markdown("#### Demographics")

    school = st.selectbox(
        "School",
        ["GP", "MS"],
    )

    sex = st.selectbox(
        "Sex",
        ["F", "M"],
    )

    address = st.selectbox(
        "Address",
        ["U", "R"],
    )

    famsize = st.selectbox(
        "Family Size",
        ["GT3", "LE3"],
    )

    Pstatus = st.selectbox(
        "Parent Status",
        ["A", "T"],
    )


with col3:

    st.markdown("#### Education")

    Medu = st.selectbox(
        "Mother Education",
        [0, 1, 2, 3, 4],
        index=2,
    )

    Fedu = st.selectbox(
        "Father Education",
        [0, 1, 2, 3, 4],
        index=2,
    )

    higher = st.selectbox(
        "Wants Higher Education",
        ["yes", "no"],
    )

    nursery = st.selectbox(
        "Attended Nursery",
        ["yes", "no"],
    )


# ============================================================
# FAMILY & SOCIAL
# ============================================================

st.markdown(
    '<div class="section-title">🏠 Family & Social Information</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)


with col1:

    Mjob = st.selectbox(
        "Mother's Job",
        ["teacher", "health", "services", "at_home", "other"],
    )

    Fjob = st.selectbox(
        "Father's Job",
        ["teacher", "health", "services", "at_home", "other"],
    )

    guardian = st.selectbox(
        "Guardian",
        ["mother", "father", "other"],
    )

    reason = st.selectbox(
        "Reason for Choosing School",
        ["course", "home", "reputation", "other"],
    )


with col2:

    schoolsup = st.selectbox(
        "Extra School Support",
        ["yes", "no"],
    )

    famsup = st.selectbox(
        "Family Educational Support",
        ["yes", "no"],
    )

    paid = st.selectbox(
        "Extra Paid Classes",
        ["yes", "no"],
    )

    activities = st.selectbox(
        "Extra-Curricular Activities",
        ["yes", "no"],
    )


with col3:

    internet = st.selectbox(
        "Internet Access",
        ["yes", "no"],
    )

    romantic = st.selectbox(
        "Romantic Relationship",
        ["yes", "no"],
    )

    traveltime = st.selectbox(
        "Travel Time",
        [1, 2, 3, 4],
        index=0,
    )

    famrel = st.slider(
        "Family Relationship",
        1,
        5,
        4,
    )


# ============================================================
# LIFESTYLE
# ============================================================

st.markdown(
    '<div class="section-title">⚡ Lifestyle & Daily Habits</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Study Time",
        f"{studytime}/4",
    )


with col2:

    freetime = st.slider(
        "Free Time",
        1,
        5,
        3,
    )


with col3:

    goout = st.slider(
        "Going Out",
        1,
        5,
        3,
    )


with col4:

    health = st.slider(
        "Health",
        1,
        5,
        3,
    )


col1, col2 = st.columns(2)


with col1:

    Dalc = st.slider(
        "Workday Alcohol Consumption",
        1,
        5,
        1,
    )


with col2:

    Walc = st.slider(
        "Weekend Alcohol Consumption",
        1,
        5,
        1,
    )


# ============================================================
# PREDICTION
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🚀  PREDICT STUDENT PERFORMANCE",
    use_container_width=True,
)


if predict_button:

    student = {
        "age": age,
        "Medu": Medu,
        "Fedu": Fedu,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "school": school,
        "sex": sex,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
    }

    try:

        prediction = predict_student(student)

        percentage = (prediction / 20) * 100

        if prediction >= 15:
            category = "High Performance"
        elif prediction >= 10:
            category = "Moderate Performance"
        else:
            category = "Lower Performance"

        st.markdown(
            f"""
<div class="result-card">
<div class="result-label">PREDICTED FINAL GRADE</div>
<div class="result-score">{prediction:.2f} / 20</div>
<div class="result-category">{category}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.progress(
            min(prediction / 20, 1.0)
        )

        st.markdown("### 📊 Prediction Summary")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Predicted G3",
                f"{prediction:.2f}",
            )

        with result_col2:

            st.metric(
                "Equivalent Percentage",
                f"{percentage:.1f}%",
            )

        with result_col3:

            st.metric(
                "Performance Level",
                category,
            )

        st.caption(
            "Prediction generated by the tuned Random Forest regression "
            "pipeline trained on the UCI Student Performance dataset."
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
Student Performance AI • Machine Learning Project<br>
UCI Student Performance Dataset • Tuned Random Forest Regression
</div>
""",
    unsafe_allow_html=True,
)