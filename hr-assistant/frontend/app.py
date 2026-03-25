import streamlit as st
import requests
import json
import os

API_URL = "http://localhost:8000"

st.set_page_config(page_title="HR Recruitment Assistant", page_icon="🤖", layout="wide")

# Custom CSS for modern look
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🤖 HR Recruitment Assistant</h1>", unsafe_allow_html=True)
st.markdown("Automate your hiring workflow with AI-powered resume screening and policy queries.")

# Sidebar
# with st.sidebar:
#     st.header("🗂️ Candidate Management")
    
#     st.subheader("1. Upload Resumes")
#     uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)
    
#     if uploaded_files:
#         if st.button("Process Resumes", use_container_width=True):
#             with st.spinner("Uploading to Server..."):
#                 paths = []
#                 for uploaded_file in uploaded_files:
#                     files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
#                     try:
#                         res = requests.post(f"{API_URL}/upload-resume/", files=files)
#                         if res.status_code == 200:
#                             paths.append(res.json().get("file_path"))
#                         else:
#                             st.error(f"Error for {uploaded_file.name}: {res.text}")
#                     except Exception as e:
#                         st.error(f"Connection failed for {uploaded_file.name}: {e}")
                
#                 if paths:
#                     st.success(f"{len(paths)} Resumes uploaded successfully!")
#                     st.session_state["resume_paths"] = paths

#     st.divider()
    
#     st.subheader("2. Job Description")
#     job_desc = st.text_area("Paste the Job Description here:", height=150)
#     if st.button("Analyze Candidates", type="primary", use_container_width=True):
#         if "resume_paths" not in st.session_state or not st.session_state["resume_paths"]:
#             st.warning("Please upload at least one resume first.")
#         elif not job_desc:
#             st.warning("Please provide a job description.")
#         else:
#             paths_str = "\n".join([f"- {p}" for p in st.session_state['resume_paths']])
#             prompt = (
#                 f"Please analyze and rank the following candidate resumes located at:\n{paths_str}\n"
#                 f"against the following Job Description:\n\n{job_desc}"
#             )
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             st.rerun()

#     st.divider()
#     st.markdown("*Note: The agent can access uploaded files and HR policies via tools.*")

# Create Tabs mapping to original Chat Interface and New ATS Dashboard
tab1, tab2 = st.tabs(["💬 Chatbot Support", "📊 ATS Dashboard"])

# ----------------- TAB 1: Chat Interface -----------------
with tab1:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I'm your HR Assistant. How can I help you today?"}]
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = "session_" + os.urandom(4).hex()

    # Display chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            # Beautify display logic so the massive prompt isn't displayed literally
            content = msg["content"]
            if "against the following Job Description" in content:
                content = "🔍 Please analyze the uploaded resume(s) against the provided Job Description."
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(msg["content"])

    # Process user input
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        user_msg = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    payload = {"message": user_msg, "thread_id": st.session_state.thread_id}
                    res = requests.post(f"{API_URL}/chat", json=payload)
                    
                    if res.status_code == 200:
                        response_content = res.json().get("response", "Error getting response.")
                        st.write(response_content)
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                    else:
                        error_msg = f"API Error {res.status_code}: {res.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"Connection failed. Make sure the FastAPI backend is running! Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Input box
    if prompt := st.chat_input("Ask about policies, candidate qualifications, or hiring guidelines..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

# ----------------- TAB 2: ATS Dashboard -----------------
with tab2:
    st.header("🧠 Resume Screening & ATS System")
    
    dash_uploaded_files = st.file_uploader("Upload Resumes (PDF) for Screening", type="pdf", key="dash_uploader", accept_multiple_files=True)
    dash_jd_text = st.text_area("Enter Job Description", key="dash_jd")

    if dash_uploaded_files:
        st.success(f"{len(dash_uploaded_files)} File(s) selected!")
        if st.button("Upload & Process Resumes"):
            with st.spinner("Processing..."):
                for uploaded_file in dash_uploaded_files:
                    response = requests.post(
                        f"{API_URL}/upload-resume/",
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    )
                    if response.status_code == 200:
                        st.subheader(f"Parsed Resume: {uploaded_file.name}")
                        st.json(response.json())
                    else:
                        st.error(f"Error parsing {uploaded_file.name}: {response.text}")

    if dash_jd_text:
        if st.button("Run ATS Scoring on Database"):
            with st.spinner("Scoring Candidates..."):
                response = requests.post(
                    f"{API_URL}/ats-score/",
                    data={"jd_text": dash_jd_text}
                )
                if response.status_code == 200:
                    st.subheader("ATS Results")
                    st.json(response.json())
                else:
                    st.error(response.text)

        if st.button("Get Best Candidate from Database"):
            with st.spinner("Finding Best Fit..."):
                response = requests.post(
                    f"{API_URL}/best-candidate/",
                    data={"jd_text": dash_jd_text}
                )
                if response.status_code == 200:
                    st.subheader("Best Candidate")
                    st.write(response.json()["result"])
                else:
                    st.error(response.text)

    if dash_uploaded_files and dash_jd_text:
        if st.button("Batch Screen & Find Best"):
            with st.spinner("Processing all resumes and finding the absolute best match..."):
                files = [("files", (file.name, file, "application/pdf")) for file in dash_uploaded_files]
                data = {"jd_text": dash_jd_text}
                
                response = requests.post(
                    f"{API_URL}/batch-screening/",
                    files=files,
                    data=data
                )
                if response.status_code == 200:
                    st.subheader("🏆 Best Candidate Analysis")
                    st.write(response.json()["result"])
                    
                    with st.expander("View Raw Parsed Data"):
                        st.json(response.json()["parsed_resumes"])
                else:
                    st.error(response.text)