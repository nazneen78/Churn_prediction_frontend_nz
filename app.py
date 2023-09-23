import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LogisticRegression
import joblib
from io import StringIO
import streamlit_ext as ste

st.set_page_config(
        page_title="Churn Prediction App",
        page_icon=":bar_chart",
        layout="wide"
    )
st.title('Churn Prediction App')


package = joblib.load("model.pkl")
loaded_model = package.named_steps['classifier']
loaded_preproc = package.named_steps['preprocessor']

def main():
    

    # upload CSV file
    st.sidebar.header('User Input')
    uploaded_file = st.file_uploader("Upload a csv file here", type=["csv"])

    st.sidebar.markdown("""
    **Instructions:**
    1. Upload a CSV file containing the data you want to predict.
    2. The file should have the same columns as the training data.
    3. After uploading, click the 'Predict' button to see predictions.

    """)
    def categorize_churn_risk(churn_probabilities):
        # Categorize users based on predicted churn percentages
        risk_categories = []
        for prob in churn_probabilities:
            if prob >= 90:
                risk_categories.append("High Risk")
            elif prob >= 50:
                risk_categories.append("Medium Risk")
            else:
                risk_categories.append("Low Risk")
        return risk_categories



    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # test with 10 rows-- to delete later
        ids= df.msno
        X_test= df.drop(['msno', 'is_churn', 'bd','payment_method_id', 'city', 'registered_via'], axis=1)

        st.sidebar.header("Please filter here: ")

        if st.button("Predict"):

            X_columns = X_test.columns.to_list()
            # pre_processor = predict_pipeline()
            X_transformed = loaded_preproc.fit_transform(X_test)
            X_transformed = pd.DataFrame(X_transformed,columns=X_columns)

            # # uploaded data
            # st.subheader("Uploaded Data:")
            # st.write(df)

            # make predictions
            predict = loaded_model.predict_proba(X_transformed)*100
            pred = predict[:,1].astype(float)
            new = pd.DataFrame({'ID': ids, 'Prediction percentage': np.round(predict[:,1],2)})

            st.subheader("predictions:")
            st.table(new.head(10))

            st.subheader("Churn Statistics")

            churn_count = new['Prediction percentage'].apply(lambda x: 'Churn' if x >= 50 else 'No Churn')
            churn_counts = churn_count.value_counts()

            fig = px.pie(new, values=churn_counts.values, names=churn_counts.index, title='Churn Distribution')
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                pull=[0.1, 0.1],
                hole=0.3,
                )
            fig.update_layout(
                    margin=dict(l=0, r=0, b=0, t=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                )
            st.plotly_chart(fig)


            churn_risks = categorize_churn_risk(new['Prediction percentage'])

            # Create a DataFrame with user IDs and their corresponding churn risks
            risk_df = pd.DataFrame({'ID': ids, 'Churn Risk': churn_risks})

            high_risk_df = risk_df[risk_df['Churn Risk'] == 'High Risk']
            medium_risk_df = risk_df[risk_df['Churn Risk'] == 'Medium Risk']
            low_risk_df = risk_df[risk_df['Churn Risk'] == 'Low Risk']

            # Display risk categories in three separate columns
            st.subheader("Churn Risk Categories:")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.subheader("High Risk")
                st.table(high_risk_df["ID"].head(10))
                col_1 = pd.DataFrame(high_risk_df["ID"]).to_csv(index=False).encode('utf-8')

                ste.download_button(
                    label="Download data as CSV",
                    data=col_1,
                    file_name='High_risk_df.csv',
                    mime='text/csv',
                )

            with col2:
                st.subheader("Medium Risk")
                st.table(medium_risk_df["ID"].head(10))
                col_2 = pd.DataFrame(medium_risk_df["ID"]).to_csv(index=False).encode('utf-8')

                ste.download_button(
                    label="Download data as CSV",
                    data=col_2,
                    file_name='Medium_risk_df.csv',
                    mime='text/csv',
                )

            with col3:
                st.subheader("Low Risk")
                st.table(low_risk_df["ID"].head(10))
                col_3 = pd.DataFrame(low_risk_df["ID"]).to_csv(index=False).encode('utf-8')

                ste.download_button(
                    label="Download data as CSV",
                    data=col_3,
                    file_name='Low_risk_df.csv',
                    mime='text/csv',
                )


        # # # Hide streamlit style

        # hide_st_style = """
        # <style>
        # #MainMenu {visibility:hidden;}
        # footer {visibility:hidden;}
        # header {visibility:hidden;}
        # </style>"""

        # st.markdown(hide_st_style, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
