import streamlit as st
import pandas as pd
from transformers import pipeline
from datetime import datetime

# Load model
pipe = pipeline("sentiment-analysis")

# Emoji map
emoji_map = {
    "positive": "😄",
    "negative": "😢",
    "neutral": "😐"
}

# App layout
st.title("🧘 Mood Checker: Daily Journal")
st.write("Write about your day, and we'll analyze your mood.")

# User input
entry = st.text_area("Write your journal entry here...", height=200)

if st.button("Analyze My Mood"):
    if entry.strip():
        result = pipe(entry[:1000])[0]
        mood = result['label'].lower()
        emoji = emoji_map.get(mood, "😐")
        st.subheader(f"Your mood: **{mood.capitalize()}** {emoji}")
        st.caption(f"Confidence: {result['score']:.2f}")

        # Save to file
        today = datetime.today().strftime("%Y-%m-%d")
        new_entry = pd.DataFrame([[today, entry, mood]], columns=["date", "entry", "mood"])

        try:
            old = pd.read_csv("journal_log.csv")
            combined = pd.concat([old, new_entry], ignore_index=True)
        except FileNotFoundError:
            combined = new_entry

        combined.to_csv("journal_log.csv", index=False)
    else:
        st.warning("Please write something first.")

# Show mood history
if st.checkbox("📈 Show mood history"):
    try:
        df = pd.read_csv("journal_log.csv")
        st.line_chart(df["mood"].apply(lambda x: 1 if x == "positive" else -1 if x == "negative" else 0))
        st.write(df.tail())
    except FileNotFoundError:
        st.info("No entries logged yet.")
