
import streamlit as st
import pandas as pd
import os
import joblib
import math

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="IoT Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🌐 IoT Analytics Dashboard")

st.write(
    "Smart Monitoring, Environmental Analysis "
    "and Industrial Predictive Maintenance"
)

# ==================================================
# 2. CACHED DATA AND MODEL LOADING
# ==================================================

@st.cache_data
def load_csv(file_path, columns=None):
    return pd.read_csv(file_path, usecols=columns)


@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)


# ==================================================
# 3. CHART DATA REDUCTION
# ==================================================

def reduce_data(df, max_points=1000):

    if len(df) > max_points:

        step = math.ceil(len(df) / max_points)

        return df.iloc[::step]

    return df


# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.selectbox(
    "Select Use Case",
    [
        "Overview",
        "Smart Home Energy",
        "Environmental Monitoring",
        "Industrial Predictive Maintenance"
    ]
)


# ==================================================
# 5. OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Use Cases", "3")

    col2.metric("ML Models", "3")

    col3.metric("Platform", "IoT Analytics")

    st.divider()

    st.subheader("Available Modules")

    st.write("🏠 Smart Home Energy Monitoring")

    st.write("🌱 Environmental Monitoring")

    st.write("🏭 Industrial Predictive Maintenance")

    st.info(
        "Select a use case from the sidebar "
        "to explore its features and predictions."
    )


# ==================================================
# 6. SMART HOME ENERGY MONITORING
# ==================================================

elif page == "Smart Home Energy":

    st.header("🏠 Smart Home Energy Monitoring")

    file = "outputs/cleaned_smart_home.csv"

    model_file = "models/energy_model.pkl"

    if os.path.exists(file):

        columns = [
            "Energy Consumption (kWh)",
            "Appliance Type",
            "Season"
        ]

        df = load_csv(file, columns)

        st.subheader("Historical Energy Analysis")

        col1, col2 = st.columns(2)

        col1.metric(
            "Average Energy Consumption",
            f"{df['Energy Consumption (kWh)'].mean():.3f} kWh"
        )

        col2.metric(
            "Total Records",
            f"{len(df):,}"
        )

        st.subheader("Historical Energy Chart")

        chart_df = reduce_data(
            df[["Energy Consumption (kWh)"]]
        )

        st.line_chart(chart_df)

        st.divider()

        st.subheader("🔮 Predict Energy Consumption")

        if os.path.exists(model_file):

            with st.form("energy_form"):

                appliance = st.selectbox(
                    "Select Appliance",
                    sorted(
                        df["Appliance Type"]
                        .dropna()
                        .unique()
                    )
                )

                temperature = st.number_input(
                    "Outdoor Temperature (°C)",
                    value=30.0
                )

                season = st.selectbox(
                    "Select Season",
                    sorted(
                        df["Season"]
                        .dropna()
                        .unique()
                    )
                )

                household = st.number_input(
                    "Household Size",
                    min_value=1,
                    value=4
                )

                date = st.date_input("Select Date")

                time = st.time_input("Select Time")

                submitted = st.form_submit_button(
                    "Predict Energy Consumption"
                )

            if submitted:

                try:

                    saved_data = load_model(model_file)

                    model = saved_data["model"]

                    model_columns = saved_data["columns"]

                    input_data = pd.DataFrame([{

                        "Outdoor Temperature (°C)": temperature,

                        "Household Size": household,

                        "Day": date.day,

                        "Month": date.month,

                        "Hour": time.hour

                    }])

                    input_data[
                        "Appliance Type_" + appliance
                    ] = 1

                    input_data[
                        "Season_" + season
                    ] = 1

                    input_data = input_data.reindex(
                        columns=model_columns,
                        fill_value=0
                    )

                    prediction = model.predict(input_data)[0]

                    st.success(
                        f"Predicted Energy Consumption: "
                        f"{prediction:.3f} kWh"
                    )

                except Exception as e:

                    st.error(f"Prediction error: {e}")

        else:

            st.warning("Energy model not found.")


# ==================================================
# 7. ENVIRONMENTAL MONITORING
# ==================================================

elif page == "Environmental Monitoring":

    st.header("🌱 Environmental Monitoring")

    file = "outputs/environment_anomaly_results_small.csv"

    model_file = "models/environment_anomaly_model.pkl"

    if (
        os.path.exists(file)
        and os.path.exists(model_file)
    ):

        columns = [
            "Status",
            "temp",
            "co",
            "lpg",
            "smoke",
            "humidity"
        ]

        df = load_csv(file, columns)

        st.subheader("Historical Environmental Analysis")

        normal = (df["Status"] == "Normal").sum()

        unusual = (df["Status"] == "Unusual").sum()

        col1, col2 = st.columns(2)

        col1.metric(
            "Normal Readings",
            f"{normal:,}"
        )

        col2.metric(
            "Unusual Readings",
            f"{unusual:,}"
        )

        st.divider()

        st.subheader("Enter Environmental Sensor Values")

        st.write(
            "Enter the sensor readings to analyze "
            "environmental conditions."
        )

        with st.form("environment_form"):

            temperature = st.number_input(
                "Temperature (°C)",
                value=float(df["temp"].median())
            )

            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(df["humidity"].median())
            )

            co = st.number_input(
                "CO level",
                min_value=0.0,
                value=float(df["co"].median())
            )

            lpg = st.number_input(
                "LPG level",
                min_value=0.0,
                value=float(df["lpg"].median())
            )

            smoke = st.number_input(
                "Smoke level",
                min_value=0.0,
                value=float(df["smoke"].median())
            )

            submitted = st.form_submit_button(
                "Analyze Environmental Data"
            )

        if submitted:

            try:

                saved_data = load_model(model_file)

                model = saved_data["model"]

                features = saved_data["features"]

                input_data = pd.DataFrame([{

                    "co": co,

                    "humidity": humidity,

                    "lpg": lpg,

                    "smoke": smoke,

                    "temp": temperature

                }])

                input_data = input_data[features]

                prediction = model.predict(input_data)[0]

                st.subheader("Environmental Analysis Result")

                if prediction == 1:

                    st.success(
                        "NORMAL: Reading follows the "
                        "model's learned normal pattern."
                    )

                else:

                    st.warning(
                        "UNUSUAL: Potential sensor anomaly detected."
                    )

            except Exception as e:

                st.error(f"Analysis error: {e}")

        st.divider()

        st.subheader("Historical Sensor Charts")

        chart_df = reduce_data(
            df[["temp", "co", "lpg", "smoke", "humidity"]]
        )

        st.write("Temperature")
        st.line_chart(chart_df[["temp"]])

        st.write("CO Level")
        st.line_chart(chart_df[["co"]])

        st.write("LPG Level")
        st.line_chart(chart_df[["lpg"]])

        st.write("Smoke Level")
        st.line_chart(chart_df[["smoke"]])

        st.write("Humidity")
        st.line_chart(chart_df[["humidity"]])

        st.caption(
            "Anomaly detection identifies unusual patterns. "
            "It does not certify environmental safety."
        )

    else:

        st.warning(
            "Environmental dataset or trained model not found."
        )


# ==================================================
# 8. INDUSTRIAL PREDICTIVE MAINTENANCE
# ==================================================

elif page == "Industrial Predictive Maintenance":

    st.header("🏭 Industrial Predictive Maintenance")

    file = "outputs/nasa_processed.csv"

    model_file = "models/nasa_rul_model.pkl"

    if os.path.exists(file):

        df = load_csv(
            file,
            ["Engine_ID", "Cycle", "RUL"]
        )

        st.subheader("Historical Engine Analysis")

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Engines",
            f"{df['Engine_ID'].nunique():,}"
        )

        col2.metric(
            "Average Remaining Useful Life",
            f"{df['RUL'].mean():.2f} cycles"
        )

        st.subheader("Historical Engine Records")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.divider()

        st.subheader("Enter Engine Sensor Values")

        if os.path.exists(model_file):

            try:

                saved_data = load_model(model_file)

                model = saved_data["model"]

                features = saved_data["features"]

                with st.form("nasa_prediction_form"):

                    st.write("Engine Information")

                    cycle = st.number_input(
                        "Engine Cycle",
                        min_value=1,
                        value=50
                    )

                    st.write("Operating Settings")

                    setting_1 = st.number_input(
                        "Setting 1",
                        value=0.0,
                        format="%.5f"
                    )

                    setting_2 = st.number_input(
                        "Setting 2",
                        value=0.0,
                        format="%.5f"
                    )

                    setting_3 = st.number_input(
                        "Setting 3",
                        value=0.0,
                        format="%.5f"
                    )

                    st.write("Engine Sensor Readings")

                    sensor_values = {}

                    for i in range(1, 22):

                        sensor_values[
                            f"Sensor_{i}"
                        ] = st.number_input(
                            f"Sensor {i}",
                            value=0.0,
                            format="%.5f"
                        )

                    submitted = st.form_submit_button(
                        "Predict Remaining Useful Life"
                    )

                if submitted:

                    input_values = {

                        "Cycle": cycle,

                        "Setting_1": setting_1,

                        "Setting_2": setting_2,

                        "Setting_3": setting_3,

                        **sensor_values

                    }

                    input_data = pd.DataFrame(
                        [input_values]
                    )[features]

                    prediction = model.predict(input_data)[0]

                    st.subheader("Prediction Result")

                    st.success(
                        f"Estimated Remaining Useful Life: "
                        f"{max(0, prediction):.2f} cycles"
                    )

                    st.info(
                        "This is a machine-learning estimate, "
                        "not a guaranteed maintenance date."
                    )

            except Exception as e:

                st.error(
                    f"NASA prediction error: {e}"
                )

        else:

            st.warning("NASA RUL model not found.")

        st.divider()

        st.subheader("Historical RUL Chart")

        first_engine = df["Engine_ID"].iloc[0]

        engine = df[
            df["Engine_ID"] == first_engine
        ].sort_values("Cycle")

        chart_df = reduce_data(
            engine.set_index("Cycle")[["RUL"]]
        )

        st.line_chart(chart_df)

    else:

        st.warning("NASA processed dataset not found.")


# ==================================================
# 9. FOOTER
# ==================================================

st.divider()

st.caption(
    "IoT Analytics Dashboard | Python | "
    "Machine Learning | Streamlit"
)






