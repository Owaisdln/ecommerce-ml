import streamlit as st
import pandas as pd
import os

from context import apply_context_adjustment
from emotion import apply_emotion_adjustment
from hybrid import HybridLayer
from explain import build_explanation


st.set_page_config(page_title="Intelligent Recommendation System", layout="wide")


st.markdown("""
<style>
.big-title {
    font-size: 38px;
    font-weight: bold;
    color: #00C4FF;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #1E1E1E;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">Intelligent Layered Recommendation System</div>', unsafe_allow_html=True)
st.caption("Context + Emotion + Hybrid + Explainability")


# Load Dataset

DATA_PATH = "review.csv"

if not os.path.exists(DATA_PATH):
    st.error("Dataset review.csv not found.")
    st.stop()

df = pd.read_csv(DATA_PATH)
df = df[['reviews.username', 'asins', 'name', 'reviews.rating']]
df = df.dropna()
df.columns = ['user_id', 'product_id', 'product_name', 'rating']

#sidebar controls
st.sidebar.header("Personalization Controls")

budget = st.sidebar.selectbox(
    "Budget",
    ["None", "Low", "Medium", "High", "Premium"]
)

occasion = st.sidebar.selectbox(
    "Occasion",
    ["None", "Casual", "Party", "Work", "Travel", "Study", "Gift", "Festival"]
)

time = st.sidebar.selectbox(
    "Time",
    ["None", "Morning", "Afternoon", "Evening", "Night", "Weekend"]
)

emotion_text = st.sidebar.text_input("Mood Input")

context_selection = {
    "budget": budget if budget != "None" else None,
    "occasion": occasion if occasion != "None" else None,
    "time": time if time != "None" else None
}

#
st.subheader("Browse Products")

product_mapping = df[['product_id', 'product_name']].drop_duplicates()
product_dict = dict(zip(product_mapping['product_id'], product_mapping['product_name']))
product_list = product_mapping['product_name'].tolist()

selected_product_name = st.selectbox("Select a Product", product_list)
selected_product_id = product_mapping[
    product_mapping['product_name'] == selected_product_name
]['product_id'].values[0]


if st.button("Generate Recommendations"):

   
    hybrid_layer = HybridLayer(df[['product_id', 'product_name']])

    cf_predictions = []

    for item in product_mapping['product_id']:
        content_score = hybrid_layer.get_content_score(item, selected_product_id)
        cf_predictions.append((item, content_score))

    cf_predictions.sort(key=lambda x: x[1], reverse=True)
    cf_predictions = cf_predictions[1:20]  # skip itself

    
    context_adjusted = apply_context_adjustment(
        cf_predictions,
        context_selection
    )

    
    emotion_adjusted = apply_emotion_adjustment(
        context_adjusted,
        emotion_text,
        product_dict
    )

    
    final_predictions = hybrid_layer.apply_hybrid(
        emotion_adjusted,
        selected_product_id
    )

    
    st.subheader("Top Personalized Recommendations")

    for product_id, final_score, context_bonus, emotion_bonus, content_score in final_predictions[:5]:

        explanation = build_explanation(
            product_id,
            context_bonus,
            emotion_bonus,
            content_score
        )

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)

            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"### {product_dict.get(product_id, product_id)}")
                st.markdown(f"Final Score: {round(final_score, 4)}")

                score_percent = min(final_score * 20, 100)
                st.progress(score_percent / 100)

                with st.expander("Explanation"):
                    st.write(explanation)

            with col2:
                st.metric("Context Boost", round(context_bonus, 3))
                st.metric("Emotion Boost", round(emotion_bonus, 3))
                st.metric("Content Similarity", round(content_score, 3))

            st.markdown('</div>', unsafe_allow_html=True)