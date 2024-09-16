import streamlit as st
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
import re 
import os
import base64
from langchain_community.chat_models import ChatOpenAI
import openai
st.set_page_config(layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# def load_css():
#     with open(r"static/styles.css", "r") as f:
#         css = f"<style>{f.read()}</style>"
#         st.markdown(css, unsafe_allow_html=True)
# load_css()
def load_css():
    try:
        with open(r"static/styles.css", "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found.")
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
            background-color: white;
            z-index: 999;
            text-align: center;
        }
        .fixed-header {
            border-bottom: 0;
        }
        .st-emotion-cache-1jicfl2 {
        padding-left: 1rem;
        padding-right: 1rem;
        margin-top: -100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ####Header and logo
# def img_to_base64(img_path):
#     with open(img_path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode()

# #  Path to your image
# img_path = "static/CC_logo.png"
# img_base64 = img_to_base64(img_path)
 
# # Create header container
# header = st.container()
# header.write(f"""
#     <div class='fixed-header'>
#         <img src="data:image/png;base64,{img_base64}" class="logo">

#     </div>
# """, unsafe_allow_html=True)

@st.cache_data
# Function to read the COBOL to Java mappings from a file
# def load_code_mappings(file_path):
#     with open(file_path, 'r') as file:
#         content = file.read()
    
#     cobol_java_pairs = re.findall(r"COBOL:(.*?)JAVA:(.*?)(?=\nCOBOL:|\Z)", content, re.DOTALL)
#     code_mappings = {cobol.strip(): java.strip() for cobol, java in cobol_java_pairs}
#     return code_mappings
def load_code_mappings(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        cobol_java_pairs = re.findall(r"COBOL:(.*?)JAVA:(.*?)(?=\nCOBOL:|\Z)", content, re.DOTALL)
        code_mappings = {cobol.strip(): java.strip() for cobol, java in cobol_java_pairs}
        return code_mappings
    except FileNotFoundError:
        st.warning("Mapping file not found.")
        return {}       
def generate_code_description(language, code, is_source=False):
    print(f"Generating {'source' if is_source else 'target'} code description for {language}", '\n\n')
    
    prompt = f"""
    Analyze the following {language} code and provide a detailed description covering the following aspects:

    1. Code Overview:
    - Briefly summarize what the code does in 2-3 sentences.

    2. Required Packages:
    - List all the packages or libraries that need to be installed to run this code.
    - For each package, provide a brief explanation of its purpose in this context.

    3. Major Features:
    - Identify and explain the main features or functionalities implemented in the code.
    - Highlight any important algorithms, data structures, or design patterns used.

    4. Code Structure:
    - Describe the overall structure of the code (e.g., classes, functions, modules).
    - Explain how different parts of the code interact with each other.

    5. Key Components:
    - List and briefly explain the most important variables, functions, or classes in the code.
    - Highlight any critical sections of the code that are essential to its functionality.

    6. Input/Output:
    - Describe what inputs the code expects and in what format.
    - Explain what outputs the code produces and how they are presented.

    7. Error Handling:
    - Mention any error handling or exception management implemented in the code.

    8. Potential Use Cases:
    - Suggest 2-3 practical applications or scenarios where this code could be useful.

    9. Limitations or Considerations:
    - Mention any limitations of the current implementation or important considerations for users.

    10. Improvement Suggestions:
        - Provide 1-2 suggestions for how the code could be improved or extended.

    Use bullet points for clarity. Focus on the most important aspects of the code.

    {language} Code:
    {code}

    Concise Code Description:
    """
    
    response = openai.OpenAI().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes and describes code."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    description = response.choices[0].message.content.strip()
    return description

def compare_and_score_models(source_code, source_description, generated_codes, target_descriptions):
    scores = [0] * 3
    explanations = []
    
    # Compare code length and complexity
    code_lengths = [len(code.split('\n')) for code in generated_codes]
    avg_length = sum(code_lengths) / 3
    for i, length in enumerate(code_lengths):
        if length < avg_length * 0.8:
            scores[i] += 1
            explanations.append(f"Model {i+1} generated more concise code.")
        elif length > avg_length * 1.2:
            explanations.append(f"Model {i+1} generated longer code, which might be more comprehensive or less efficient.")
    
    # Compare explanation quality
    explanation_lengths = [len(desc.split()) for desc in target_descriptions]
    for i, length in enumerate(explanation_lengths):
        if length > 50:
            scores[i] += 1
            explanations.append(f"Model {i+1} provided a more detailed explanation.")
    
    # Check for specific COBOL to target language patterns
    cobol_patterns = ['PERFORM', 'MOVE', 'COMPUTE']
    java_patterns = ['for', 'while', '=', 'Math.']
    
    for i, code in enumerate(generated_codes):
        java_matches = sum(1 for pattern in java_patterns if pattern in code)
        cobol_matches = sum(1 for pattern in cobol_patterns if pattern in code)
        if java_matches >= 2 and cobol_matches == 0:
            scores[i] += 1
            explanations.append(f"Model {i+1} correctly translated common COBOL constructs to Java.")
        elif cobol_matches > 0:
            explanations.append(f"Model {i+1} retained some COBOL syntax, which may indicate incomplete translation.")
    
    # Analyze code structure
    for i, code in enumerate(generated_codes):
        if re.search(r'public\s+class', code) and re.search(r'public\s+static\s+void\s+main', code):
            scores[i] += 1
            explanations.append(f"Model {i+1} generated a proper Java class structure with a main method.")
    
    # Check for exception handling
    for i, code in enumerate(generated_codes):
        if 'try' in code and 'catch' in code:
            scores[i] += 1
            explanations.append(f"Model {i+1} implemented exception handling.")
    
    # Analyze variable naming conventions
    for i, code in enumerate(generated_codes):
        camelCase = len(re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b', code))
        if camelCase > 5:
            scores[i] += 1
            explanations.append(f"Model {i+1} used proper Java camelCase naming conventions.")
    
    best_model = scores.index(max(scores)) + 1
    
    return best_model, scores, explanations

# LLM function to provide detailed analysis
def llm_analyze_comparison(source_code, source_description, generated_codes, target_descriptions, best_model, scores, explanations):
    try:
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        # huggingfacehub_api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
        huggingfacehub_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, huggingfacehub_api_token=huggingfacehub_api_token, max_new_tokens=4096, timeout=600)

        prompt = f"""
        As an expert in COBOL to Java conversion and software engineering, provide a detailed analysis of the best performing model for COBOL to Java conversion based on the following comparison results:

        Source COBOL code snippet:
        {source_code[:200]}...

        Source code description:
        {source_description[:200]}...

        Comparison results:
        Best Model (according to scores): Model {best_model}
        Scores: {scores}
        Explanations: {explanations}

        Please provide a comprehensive analysis of the best model (Model {best_model}) in the following format:

        1. Best Model Overview:
            - Confirm that Model {best_model} is indeed the best performer.
            - State its overall score and how it compares to the other models.

        2. Key Strengths of Model {best_model}:
            - Provide a bullet-point list of the model's main strengths, as indicated by the comparison results.
            - For each strength, explain its significance in the context of COBOL to Java conversion.

        3. Specific Advantages:
            - List and elaborate on any specific advantages Model {best_model} demonstrated, such as:
                * Code conciseness or comprehensiveness
                * Quality of code explanations
                * Correct translation of COBOL constructs
                * Proper Java structure and conventions
                * Exception handling
                * Naming conventions
            - For each advantage, provide a brief explanation of why it's important for effective COBOL to Java conversion.

        4. Comparative Analysis:
            - In bullet points, highlight how Model {best_model} outperformed the other models in specific areas.
            - Mention any unique features or approaches that set Model {best_model} apart.

        5. Potential Areas for Improvement:
            - If applicable, list any areas where Model {best_model} could still improve, based on the comparison results.

        6. Conclusion:
            - Summarize why Model {best_model} is the best choice for COBOL to Java conversion based on this analysis.

        Please ensure your analysis is detailed, using bullet points for clarity, and focuses solely on the information provided by the 'compare_and_score_models' function. Provide concrete examples from the comparison results where possible.
        """

        response = llm(prompt)
        if not response:
            st.warning("Received empty response from the LLM for analysis.")
        return response
    except Exception as e:
        st.error(f"Error during LLM comparison analysis: {e}")
        return ""



# Define functions for each model code
def model1(source_language, target_language, source_code, code_mappings):
    source_description = generate_code_description(source_language, source_code)
    if source_language == "COBOL" and source_code.strip() in code_mappings:
        generated_code = code_mappings[source_code.strip()]
    
    llm2 = ChatOpenAI(model='gpt-4o',api_key=st.secrets["OPENAI_API_KEY"])
    prompt = generate_prompt(source_language, target_language, source_code)
    output_code = llm2.invoke(prompt).content
    if "```" in output_code: 
        output_code = output_code.split("```java")[1].split("```")[0]
    generated_code = extract_target_language_code(output_code, target_language)
    target_description = generate_code_description(target_language, generated_code)
    execution_result = ""
    # if target_language.lower() == "java":
    #     execution_result = save_and_execute_java(generated_code)
    
    # Generate microservices implementation
    # microservices_implementation = generate_microservices_implementation(target_language, generated_code, source_code, generated_code)
    # microservices_explanation = explain_microservices(microservices_implementation)
    
    return generated_code, source_description, target_description, execution_result

def model2(source_language, target_language, source_code):
    source_description = generate_code_description(source_language, source_code)
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    huggingfacehub_api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, huggingfacehub_api_token=huggingfacehub_api_token, max_new_tokens=4096, timeout=300)
    
    prompt = generate_prompt(source_language, target_language, source_code)
    # print(prompt)
    output_code = llm(prompt)
    if "```" in output_code: 
        output_code=output_code.split("```java")[1].split("```")[0]
    generated_code = extract_target_language_code(output_code, target_language)
    target_description = generate_code_description(target_language, generated_code)
    return generated_code, source_description,target_description


def model3(source_language, target_language, source_code):
    source_description = generate_code_description(source_language, source_code)
    repo_id = "google/gemma-2b"
    huggingfacehub_api_token = st.secrets["HUGGINGFACE_API_TOKEN"]
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, huggingfacehub_api_token=huggingfacehub_api_token, max_new_tokens=4096, timeout=300)
    prompt = generate_prompt(source_language, target_language, source_code)
    output_code = llm(prompt)
    cleaned_output_code = remove_code_fences(output_code)
    cleaned_code = clean_output_code(cleaned_output_code)
    if "```" in output_code: 
        output_code=output_code.split("```java")[1].split("```")[0]
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

source_description = ""
if st.button("Convert"):
        # Generate source description once
    with st.spinner("Analyzing source code..."):
        source_description = generate_code_description(source_language, source_code)
    
    # if show_descriptions:
    with st.expander("Source_Code"):
        st.subheader("Source Code Description")
        st.markdown(source_description)

    col1, col2, col3 = st.columns(3)
    generated_codes = []
    target_descriptions = []

    with col1:
        with st.expander("Model 1"):
            st.subheader("Model 1")
            with st.spinner("Converting code and generating descriptions..."):
                generated_code1, source_desc1, target_desc1, execution_result1= model1(source_language, target_language, source_code, code_mappings)
                st.code(generated_code1, language=target_language.lower())
                # st.subheader("Source Code Description")
                # st.markdown(source_desc1)
        with st.expander("Generated Code Description"):
            st.subheader("Generated Code Description")
            st.markdown(target_desc1)
            generated_codes.append(generated_code1)
            target_descriptions.append(target_desc1)

    with col2:
        with st.expander("Model 2"):

            st.subheader("Model 2")
            with st.spinner("Converting code and generating descriptions..."):
                generated_code2, source_desc2, target_desc2 = model2(source_language, target_language, source_code)
                st.code(generated_code2, language=target_language.lower())
                # st.subheader("Source Code Description")
                # st.markdown(source_desc2)
        with st.expander("Generated Code Description"):
            st.subheader("Generated Code Description")
            st.markdown(target_desc2)
            generated_codes.append(generated_code2)
            target_descriptions.append(target_desc2)


    with col3:
         with st.expander("Model 3"):

            st.subheader("Model 3")
            with st.spinner("Converting code and generating descriptions..."):
                generated_code3, source_desc3, target_desc3 = model3(source_language, target_language, source_code)
                st.code(generated_code3, language=target_language.lower())
                # st.subheader("Source Code Description")
                # st.markdown(source_desc3)
         with st.expander("Generated Code Description"):

            st.subheader("Generated Code Description")
            st.markdown(target_desc3)
            generated_codes.append(generated_code3)
            target_descriptions.append(target_desc3)
            

    with st.expander("Model Comparsion"):
        st.subheader("Comprehensive Analysis")
        best_model, scores, explanations = compare_and_score_models(source_code, source_description, generated_codes, target_descriptions)
        with st.spinner("Performing detailed analysis of model outputs..."):
            llm_analysis = llm_analyze_comparison(source_code, source_description, generated_codes, target_descriptions, best_model, scores, explanations)
        st.markdown(llm_analysis)



