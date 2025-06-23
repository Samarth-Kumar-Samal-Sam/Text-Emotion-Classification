import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import joblib

# Load the model
model = joblib.load(r'Model\Text-Model.joblib')

# Predicting the emotion
def predict_emotions(docx):
    result = model.predict([docx])
    return result[0]

# Getting prediction probabilities
def get_prediction_proba(docx):
    result = model.predict_proba([docx])
    return result

# Updated emoji dictionary with all 8 classes
emotions_emoji_dict = {
    'anger': '😡',
    'disgust': '🤢',
    'fear': '😨',
    'joy': '😄',
    'neutral': '😐',
    'sadness': '😢',
    'shame': '😳',
    'surprise': '😲'
}

# Main Streamlit app
def main():
    # Set page config
    st.set_page_config(
        page_title='Text Emotion Classification Application',
        page_icon='💻',
        layout='wide'
    )

    st.title("Text Emotion Classifier Application")

    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("Type the statement here")
        submit_text = st.form_submit_button(label='Predict')

        if submit_text:
            col1, col2 = st.columns(2)

            # Make prediction
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "❓")
                st.write(f"{prediction}: {emoji_icon}")
                st.write(f"Confidence: {np.max(probability):.2f}")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=model.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='emotions',
                    y='probability',
                    color='emotions'
                )
                st.altair_chart(fig, use_container_width=True)

# Run the app
if __name__ == '__main__':
    main()
