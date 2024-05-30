import streamlit as st
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
import os
import re 

@st.cache_data
# Function to read the COBOL to Java mappings from a file
def load_code_mappings(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    cobol_java_pairs = re.findall(r"COBOL:(.*?)JAVA:(.*?)(?=\nCOBOL:|\Z)", content, re.DOTALL)
    code_mappings = {cobol.strip(): java.strip() for cobol, java in cobol_java_pairs}
    return code_mappings

# Define functions for each model code
def model1(source_language, target_language, source_code, code_mappings):
    if source_language == "COBOL" and source_code.strip() in code_mappings:
        return code_mappings[source_code.strip()]

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, huggingfacehub_api_token=st.secrets["HUGGINGFACE_API_TOKEN"], max_new_tokens=4096, timeout=300)

    prompt = generate_prompt(source_language, target_language, source_code)
    output_code = llm(prompt)
    generated_code = extract_target_language_code(output_code, target_language)

    return generated_code

def model2(source_language, target_language, source_code):
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    huggingfacehub_api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, huggingfacehub_api_token=huggingfacehub_api_token, max_new_tokens=2000, timeout=300)
    
    prompt = generate_prompt(source_language, target_language, source_code)
    output_code = llm(prompt)
    generated_code = extract_target_language_code(output_code, target_language)

    return generated_code

def model3(source_language, target_language, source_code):
    repo_id = "google/gemma-2b"
    huggingfacehub_api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, huggingfacehub_api_token=huggingfacehub_api_token, max_new_tokens=2000, timeout=300)

    prompt = generate_prompt(source_language, target_language, source_code)
    output_code = llm(prompt)
    cleaned_output_code = remove_code_fences(output_code)
    cleaned_code = clean_output_code(cleaned_output_code)

    return cleaned_code

# Function to generate prompt
def generate_prompt(source_language, target_language, source_code):
    prompt = f"{source_language}:\n{source_code}\n\n{target_language}:\n"
    return prompt

# Function to clean and extract the converted code from the output
def clean_output_code(output_code):
    lines = output_code.split('\n')
    code_lines = []
    comment_count = 0

    for line in lines:
        comment_count += line.count("*/")
        if comment_count == 1:
            break
        code_lines.append(line)

    cleaned_code = '\n'.join(code_lines).strip()
    return cleaned_code

# Function to remove markdown code fences
def remove_code_fences(code):
    return code.replace("```", "").strip()

# Function to extract only the target language code from the output
def extract_target_language_code(output_code, target_language):
    # Ensure target language is followed by a colon and then capture everything until a new language or end of string
    pattern = re.compile(rf"{target_language}:\s*(.*?)(?=\n(?:COBOL|Java|Python|C\+\+|C#|JavaScript):|$)", re.DOTALL)
    match = pattern.search(output_code)
    if match:
        return match.group(1).strip()
    return output_code

# Set page layout to wide mode
st.set_page_config(layout="wide")

# Title and logos
st.markdown("<h1 style='text-align: center;'>Code Conversion</h1>", unsafe_allow_html=True)

# Supported languages
languages = ["COBOL", "Java", "Python", "C++", "C#", "JavaScript"]
languages1 = ["Java", "Python", "C++", "C#", "JavaScript"]

# Input fields
source_language = st.selectbox("Select the source language", languages)
target_language = st.selectbox("Select the target language", languages1)
source_code = st.text_area("Enter the source code", height=300)

# Load code mappings
code_mappings = load_code_mappings("Cobol_to_java_conv.txt")

# Convert button
if st.button("Convert"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Model 1")
            with st.spinner("Converting code..."):
                generated_code1 = model1(source_language, target_language, source_code, code_mappings)
                st.code(generated_code1, language=target_language.lower())

        with col2:
            st.subheader("Model 2")
            with st.spinner("Converting code..."):
                generated_code2 = model2(source_language, target_language, source_code)
                st.code(generated_code2, language=target_language.lower())

        with col3:
            st.subheader("Model 3")
            with st.spinner("Converting code..."):
                generated_code3 = model3(source_language, target_language, source_code)
                st.code(generated_code3, language=target_language.lower())
