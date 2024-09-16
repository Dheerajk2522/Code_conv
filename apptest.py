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

@st.cache_data
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
        ]
    )
    
    description = response.choices[0].message.content.strip()
    return description
# def extract_class_name(java_code):
#     match = re.search(r'public\s+class\s+(\w+)', java_code)
#     if match:
#         return match.group(1)
#     return "GeneratedCode"

# def save_and_execute_java(java_code):
#     class_name = extract_class_name(java_code)
#     with tempfile.TemporaryDirectory() as tmpdir:
#         # Save the Java code to a temporary file with the correct class name
#         java_file_path = os.path.join(tmpdir, f"{class_name}.java")
#         with open(java_file_path, "w") as java_file:
#             java_file.write(java_code)
        
#         # Compile the Java code
#         compile_command = ["javac", java_file_path]
#         compile_process = subprocess.run(compile_command, capture_output=True, text=True)
        
#         if compile_process.returncode != 0:
#             return f"Compilation Error:\n{compile_process.stderr}"
#         st.write("Compilation successful")
#         return compile_process.stdout

def compare_models(source_code, source_description, generated_codes, target_descriptions):
    comparisons = []
    for i in range(3):
        prompt = f"""
        As an expert in COBOL to Java conversion, compare the following Java code conversion based on the original COBOL code:

        Original COBOL code:
        {source_code}

        Source code description:
        {source_description}

        Model {i+1} Java code:
        {generated_codes[i]}

        Model {i+1} description:
        {target_descriptions[i]}

        Please analyze this conversion based on the following criteria:
        1. Accuracy of translation
        2. Code structure and organization
        3. Java best practices and conventions
        4. Handling of COBOL-specific constructs
        5. Time Complexity
        6. Space Complexity
        7. Code efficiency

        Provide a detailed analysis of its strengths and weaknesses, paying special attention to time complexity, space complexity, and code efficiency.

        Format your response as follows:
        Model {i+1} Analysis:
        [Your analysis here, including specific comments on time complexity, space complexity, and code efficiency]
        """

        response = openai.OpenAI().chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in COBOL to Java conversion. Analyze the given conversion."},
                {"role": "user", "content": prompt}
            ]
        )
        comparisons.append(response.choices[0].message.content.strip())
    
    # Combine individual comparisons
    combined_comparison = "\n\n".join(comparisons)
    
    # Final comparison to determine the best model
    final_prompt = f"""
    Based on the following analyses of three COBOL to Java conversion models:

    {combined_comparison}

    Please determine which model performed the best overall and explain why.

    Format your response as follows:
    Best Model:
    [State which model is best]

    Reason for Selection:
    [Explain why this model is the best, considering all criteria including time complexity, space complexity, and code efficiency]
    """

    final_response = openai.OpenAI().chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in COBOL to Java conversion. Determine the best model based on the given analyses."},
            {"role": "user", "content": final_prompt}
        ]
    )
    
    return combined_comparison + "\n\n" + final_response.choices[0].message.content.strip()

def explain_comparison(comparison_result):
    
    prompt = f"""
    Based on the following comparison of three COBOL to Java conversion models:

    {comparison_result}

    Please provide a comprehensive analysis of the comparison results in the following format:

    1. Overview:
        - Summarize the key findings from the comparison.
        - Confirm which model was determined to be the best performer.

    2. Detailed Analysis:
        - For each model, provide a bullet-point list of its main strengths and weaknesses, including:
          * Accuracy of translation
          * Code structure and organization
          * Java best practices and conventions
          * Handling of COBOL-specific constructs
          * Time Complexity
          * Space Complexity
          * Code efficiency
        - Explain the significance of these points in the context of COBOL to Java conversion.

    3. Best Model Justification:
        - Elaborate on why the chosen model is considered the best.
        - Highlight the specific advantages that set it apart from the other models, particularly in terms of time complexity, space complexity, and code efficiency.

    4. Comparative Insights:
        - Discuss any notable differences between the models' approaches or outputs, especially regarding algorithmic efficiency.
        - Identify any unique features or techniques employed by each model that impact performance or resource usage.

    5. Potential Improvements:
        - Suggest areas where each model, including the best one, could potentially improve, focusing on optimizing time complexity, space complexity, and overall code efficiency.

    6. Conclusion:
        - Summarize why the best model is the most suitable choice for COBOL to Java conversion based on this analysis, emphasizing its balance of accuracy, efficiency, and best practices.
        - Provide any final thoughts or recommendations for using or further developing these conversion models to enhance their performance and resource utilization.

    Please ensure your analysis is detailed and focuses on the information provided in the comparison results.
    Use bullet points for clarity where appropriate and provide specific examples or metrics when discussing time complexity, space complexity, and code efficiency.
    """

    response = openai.OpenAI().chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes code conversion models."},
            {"role": "user", "content": prompt}
        ]
    )
    analysis = response.choices[0].message.content.strip()
    return analysis

def generate_microservices_implementation(source_language, target_language, source_code, generated_code):
    prompt = f"""
    Analyze the following {source_language} code and its corresponding {target_language} conversion:

    {source_language} Code:
    {source_code}

    {target_language} Conversion:
    {generated_code}

    Based on this analysis, create a microservices implementation in {target_language} that:

    1. Identifies and extracts the key functionalities from the original {source_language} code.
    2. Converts these functionalities into separate, independent microservices in {target_language}.
    3. Defines appropriate REST endpoints for each microservice.
    4. Ensures proper communication and data flow between the microservices.
    5. Adheres to microservices best practices and design patterns.

    Please provide:
    1. A list of the identified key functionalities from the {source_language} code.
    2. The {target_language} code for each microservice, including:
       - Service definitions
       - REST endpoints
       - Any necessary data models
       - Inter-service communication methods
    3. A brief explanation of how the microservices interact and the overall architecture.
    4. Any necessary configuration or setup code.

    You have the freedom to choose the most appropriate microservices framework or approach for {target_language}, based on the specific requirements of the converted code.
    """

    response = openai.OpenAI().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are an expert in converting {source_language} to {target_language} and implementing microservices architectures."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content.strip()

def explain_microservices(microservices_implementation):
    prompt = f"""
    Provide a detailed explanation of the following Spring Boot microservices implementation:

    {microservices_implementation}

    Include:
    1. An overview of the microservices architecture.
    2. The purpose and functionality of each microservice.
    3. How the microservices communicate with each other.
    4. Any design patterns or best practices used.
    5. Potential scalability and maintainability benefits of this approach.
    """

    response = openai.OpenAI().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in explaining microservices architectures."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content.strip()

def model1(source_language, target_language, source_code, code_mappings):
    source_description = generate_code_description(source_language, source_code)
    if source_language == "COBOL" and source_code.strip() in code_mappings:
        generated_code = code_mappings[source_code.strip()]
    
    llm = ChatOpenAI(model='gpt-4o')
    prompt = generate_prompt(source_language, target_language, source_code)
    output_code = llm.invoke(prompt).content
    if "```" in output_code: 
        output_code = output_code.split("```java")[1].split("```")[0]
    generated_code = extract_target_language_code(output_code, target_language)
    target_description = generate_code_description(target_language, generated_code)
    execution_result = ""
    microservices_implementation = generate_microservices_implementation(target_language, generated_code, source_code, generated_code)
    microservices_explanation = explain_microservices(microservices_implementation)
    
    return generated_code, source_description, target_description, execution_result, microservices_implementation, microservices_explanation

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
                generated_code1, source_desc1, target_desc1, execution_result1, microservices_impl1, microservices_expl1= model1(source_language, target_language, source_code, code_mappings)
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
        comparison_result = compare_models(source_code, source_description, generated_codes, target_descriptions)
        with st.spinner("Performing detailed analysis of model outputs..."):
            llm_analysis = explain_comparison(comparison_result)
        st.markdown(llm_analysis)
    
    # with st.expander("Java Execution Result"):
    #     if target_language.lower() == "java":
    #         execution_result = save_and_execute_java(generated_code1)

    with st.expander("Microservices Implementation and Explanation"):
            st.subheader("Microservices Implementation")
            st.code(microservices_impl1, language="java")
            st.subheader("Microservices Explanation")
            st.markdown(microservices_expl1)



