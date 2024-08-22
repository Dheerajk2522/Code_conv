import streamlit as st
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
import re 
import os

st.set_page_config(layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
header = st.container()
header.title("Code Conversion")
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

def load_css():
    with open(r"static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)
load_css()


st.markdown(
    """
    <style>
        .st-emotion-cache-vj1c9o {
            background-color:rgb(38 39 48 / 0%);
        }
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 0; /* Stick to top edge */
            background-color: rgba(230, 234, 241);
            z-index: 999;
            text-align: center;
        }
        .fixed-header {
            border-bottom: 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
# Function to read the COBOL to Java mappings from a file
def load_code_mappings(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    cobol_java_pairs = re.findall(r"COBOL:(.*?)JAVA:(.*?)(?=\nCOBOL:|\Z)", content, re.DOTALL)
    code_mappings = {cobol.strip(): java.strip() for cobol, java in cobol_java_pairs}
    return code_mappings

def generate_code_description(language, code):
    print("Generating Code",'\n\n')
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    huggingfacehub_api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, huggingfacehub_api_token=huggingfacehub_api_token, max_new_tokens=1024, timeout=300)
    
    prompt = f"Analyze the following {language} code and provide a concise bullet point description of its main features and functionality:\n\n{code}\n\nBullet point description:"
    
    description = llm(prompt)
    return description.strip()

# Define functions for each model code
def model1(source_language, target_language, source_code, code_mappings):
    source_description = generate_code_description(source_language, source_code)
    if source_language == "COBOL" and source_code.strip() in code_mappings:
        generated_code = code_mappings[source_code.strip()]
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    huggingfacehub_api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, huggingfacehub_api_token=huggingfacehub_api_token, max_new_tokens=4096, timeout=300)

    prompt = generate_prompt(source_language, target_language, source_code)
    output_code = llm(prompt)
    if "```" in output_code: 
        output_code=output_code.split("```java")[1].split("```")[0]
    generated_code = extract_target_language_code(output_code, target_language)
    target_description = generate_code_description(target_language, generated_code)
    return generated_code, source_description, target_description

def model2(source_language, target_language, source_code):
    source_description = generate_code_description(source_language, source_code)
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    huggingfacehub_api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, huggingfacehub_api_token=huggingfacehub_api_token, max_new_tokens=2000, timeout=300)
    
    prompt = generate_prompt(source_language, target_language, source_code)
    output_code = llm(prompt)
    if "```" in output_code: 
        output_code=output_code.split("```java")[1].split("```")[0]
    generated_code = extract_target_language_code(output_code, target_language)
    target_description = generate_code_description(target_language, generated_code)
    return generated_code, source_description, target_description


def model3(source_language, target_language, source_code):
    source_description = generate_code_description(source_language, source_code)
    repo_id = "google/gemma-2b"
    huggingfacehub_api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, huggingfacehub_api_token=huggingfacehub_api_token, max_new_tokens=2000, timeout=300)

    prompt = generate_prompt(source_language, target_language, source_code)
    output_code = llm(prompt)
    if "```" in output_code: 
        output_code=output_code.split("```java")[1].split("```")[0]
    cleaned_output_code = remove_code_fences(output_code)
    cleaned_code = clean_output_code(cleaned_output_code)

    target_description = generate_code_description(target_language, cleaned_code)
    return cleaned_code, source_description, target_description

# Function to generate prompt
def generate_prompt(source_language, target_language, source_code):
    prompt = f"{source_language}:\n{source_code}\nConvert the above code into the following targer language:\n{target_language}:\n"
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


# 2nd Function to extract only the targeted language code from the output
def extract_target_language_code(output_code, target_language):
    # Split the output into sections based on language headers
    sections = re.split(r'\n(?=(?:COBOL|Java|Python|C\+\+|C#|JavaScript):)', output_code)
    
    # Find the section that starts with the target language
    for section in sections:
        if section.strip().startswith(f"{target_language}:"):
            # Remove the language header and any leading/trailing whitespace
            code = re.sub(f"^{target_language}:", "", section).strip()
            return code
    
    # If no matching section is found, return the original output
    return output_code.strip() 

# Supported languages
languages = ["COBOL", "Java", "Python", "C++", "C#", "JavaScript"]
languages1 = ["Java", "Python", "C++", "C#", "JavaScript"]

# Input fields
source_language = st.selectbox("Select the source language", languages)
target_language = st.selectbox("Select the target language", languages1)
source_code = st.text_area("Enter the source code", height=300)


# Load code mappings
code_mappings = load_code_mappings("Cobol_to_java_conv.txt")



# for Discription 
# show_descriptions = st.checkbox("Show code descriptions", value=True)
source_description = ""
if st.button("Convert"):
        # Generate source description once
    with st.spinner("Analyzing source code..."):
        source_description = generate_code_description(source_language, source_code)
    
    # if show_descriptions:
    st.subheader("Source Code Description")
    st.markdown(source_description)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Model 1")
        with st.spinner("Converting code and generating descriptions..."):
            generated_code1, source_desc1, target_desc1 = model1(source_language, target_language, source_code, code_mappings)
            st.code(generated_code1, language=target_language.lower())
            st.subheader("Generated Code Description")
            st.markdown(target_desc1)

    with col2:
        st.subheader("Model 2")
        with st.spinner("Converting code and generating descriptions..."):
            generated_code2, source_desc2, target_desc2 = model2(source_language, target_language, source_code)
            st.code(generated_code2, language=target_language.lower())
            st.subheader("Generated Code Description")
            st.markdown(target_desc2)

    with col3:
        st.subheader("Model 3")
        with st.spinner("Converting code and generating descriptions..."):
            generated_code3, source_desc3, target_desc3 = model3(source_language, target_language, source_code)
            st.code(generated_code3, language=target_language.lower())
            st.subheader("Generated Code Description")
            st.markdown(target_desc3)
