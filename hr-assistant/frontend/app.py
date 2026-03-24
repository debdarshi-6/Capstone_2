import streamlit as st
import requests

st.title("🧠 Resume Screening & ATS System")

API_URL = "http://localhost:8000"

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

jd_text = st.text_area("Enter Job Description")


if uploaded_file:

    st.success("File uploaded!")

    if st.button("Upload & Process Resume"):

        response = requests.post(
            f"{API_URL}/upload-resume/",
            files={"file": uploaded_file}
        )

        if response.status_code == 200:
            st.subheader("Parsed Resume")
            st.json(response.json())
        else:
            st.error(response.text)


if jd_text:

    if st.button("Run ATS Scoring"):

        response = requests.post(
            f"{API_URL}/ats-score/",
            data={"jd_text": jd_text}
        )

        if response.status_code == 200:
            st.subheader("ATS Results")
            st.json(response.json())
        else:
            st.error(response.text)


    if st.button("Get Best Candidate"):

        response = requests.post(
            f"{API_URL}/best-candidate/",
            data={"jd_text": jd_text}
        )

        if response.status_code == 200:
            st.subheader("Best Candidate")
            st.write(response.json()["result"])
        else:
            st.error(response.text)


if uploaded_file and jd_text:

    if st.button("Full Screening"):

        response = requests.post(
            f"{API_URL}/screening/",
            files={"file": uploaded_file}
        )

        if response.status_code == 200:
            st.subheader("Full Output")
            st.json(response.json())
        else:
            st.error(response.text)