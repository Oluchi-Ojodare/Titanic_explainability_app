
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# 1. App configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic Survival Prediction")

st.write(
    """
    This app predicts whether a Titanic passenger would have survived
    using three machine-learning models.

    Enter passenger information below to compare their predictions
    and estimated survival probabilities.
    """
)

# --------------------------------------------------
# 2. Load the trained models
# --------------------------------------------------

decision_tree = joblib.load("decision_tree.joblib")
random_forest = joblib.load("random_forest.joblib")
logistic_regression = joblib.load("logistic_regression.joblib")

models = {
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Logistic Regression": logistic_regression
}

# --------------------------------------------------
# 3. Collect passenger information
# --------------------------------------------------

st.header("Passenger Information")

with st.form("passenger_form"):

    col1, col2 = st.columns(2)

    with col1:
        Pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            index=2
        )

        Sex = st.selectbox(
            "Sex",
            options=["male", "female"]
        )

        Age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=29.0
        )

        Fare = st.number_input(
            "Ticket Fare",
            min_value=0.0,
            max_value=600.0,
            value=32.0
        )

    with col2:
        SibSp = st.number_input(
            "Number of Siblings/Spouses Aboard",
            min_value=0,
            max_value=10,
            value=0
        )

        Parch = st.number_input(
            "Number of Parents/Children Aboard",
            min_value=0,
            max_value=10,
            value=0
        )

        Embarked = st.selectbox(
            "Port of Embarkation",
            options=["S", "C", "Q"],
            format_func=lambda x: {
                "S": "Southampton",
                "C": "Cherbourg",
                "Q": "Queenstown"
            }[x]
        )

    submitted = st.form_submit_button(
        "Predict Survival",
        use_container_width=True
    )

# --------------------------------------------------
# 4. Generate predictions
# --------------------------------------------------

if submitted:

    input_data = pd.DataFrame([{
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": Embarked
    }])

    st.divider()
    st.header("Prediction Results")

    results = []

    for model_name, model in models.items():

        prediction = int(model.predict(input_data)[0])

        # Locate the probability for the survived class (1).
        class_position = list(model.classes_).index(1)

        survival_probability = model.predict_proba(
            input_data
        )[0][class_position]

        results.append({
            "Model": model_name,
            "Prediction": prediction,
            "Survival Probability": survival_probability
        })

    # --------------------------------------------------
    # 5. Display individual model results
    # --------------------------------------------------

    columns = st.columns(3)

    for column, result in zip(columns, results):

        with column:
            st.subheader(result["Model"])

            if result["Prediction"] == 1:
                st.success("Predicted to Survive")
            else:
                st.error("Predicted Not to Survive")

            st.metric(
                "Survival Probability",
                f"{result['Survival Probability']:.1%}"
            )

    # --------------------------------------------------
    # 6. Compare model predictions
    # --------------------------------------------------

    st.divider()
    st.subheader("Model Comparison")

    comparison = pd.DataFrame(results)

    comparison["Prediction"] = comparison["Prediction"].map({
        0: "Did Not Survive",
        1: "Survived"
    })

    comparison["Survival Probability"] = (
        comparison["Survival Probability"].apply(
            lambda x: f"{x:.1%}"
        )
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Survival probabilities are model estimates, not guarantees "
        "of actual outcomes."
    )

# --------------------------------------------------
# 7. About the project
# --------------------------------------------------

st.divider()

with st.expander("About this application"):

    st.write(
        """
        This application is part of an explainable machine-learning
        project using the Titanic passenger dataset.

        It compares three classification algorithms:

        - Decision Tree
        - Random Forest
        - Logistic Regression

        Each model receives the same passenger information.
        Their predictions and estimated survival probabilities
        are displayed for comparison.

        This is an educational machine-learning demonstration.
        """
    )

