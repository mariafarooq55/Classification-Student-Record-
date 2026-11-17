import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model and its column list (made in Step 10 of the notebook)
model = joblib.load("student_model.pkl")
columns = joblib.load("student_model_columns.pkl")
education_numbers = {"Matric": 0, "Intermediate": 1, "Bachelors": 2, "Masters": 3}

# 2. Page title
st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Will this student pass?")
st.write("Enter the student's details and click **Predict**.")

# 3. Input boxes, in two columns
left, right = st.columns(2)
with left:
    city = st.selectbox("City", ["Sukkur", "Rohri", "Shikarpur", "Khairpur", "Larkana"])
    age = st.number_input("Age", min_value=15, max_value=40, value=21)
    education = st.selectbox("Education", ["Matric", "Intermediate", "Bachelors", "Masters"], index=1)
    study_hours = st.slider("Study hours per week", 0.0, 25.0, 5.0, step=0.5)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    quiz = st.slider("Quiz average (out of 100)", 0, 100, 58)
with right:
    assignments = st.slider("Assignments done (out of 10)", 0, 10, 6)
    travel = st.number_input("Travel time to class (minutes)", min_value=0, max_value=180, value=20)
    laptop = st.radio("Has a laptop?", ["Yes", "No"], horizontal=True)
    internet = st.radio("Has internet at home?", ["Yes", "No"], horizontal=True)
    job = st.radio("Part-time job?", ["No", "Yes"], horizontal=True)
    programming = st.radio("Learned programming before?", ["No", "Yes"], horizontal=True)

# 4. When the button is clicked: prepare the data exactly like in the notebook, then predict
if st.button("Predict", type="primary"):
    student = pd.DataFrame([{
        "age": age,
        "education": education_numbers[education],
        "study_hours_week": study_hours,
        "attendance_pct": attendance,
        "quiz_avg": quiz,
        "assignments_done": assignments,
        "has_laptop": 1 if laptop == "Yes" else 0,
        "has_internet": 1 if internet == "Yes" else 0,
        "part_time_job": 1 if job == "Yes" else 0,
        "programming_before": 1 if programming == "Yes" else 0,
        "travel_time_min": travel,
        "city_" + city: 1,
    }])
    student = student.reindex(columns=columns, fill_value=0)
    chance = model.predict_proba(student)[0, 1]

    st.progress(float(chance), text=f"Chance to pass: {chance:.0%}")
    if chance >= 0.5:
        st.success(f"✅ PASS — {chance:.0%} chance to pass")
    else:
        st.error(f"⚠️ AT RISK — only {chance:.0%} chance to pass. Please give this student extra help!")
