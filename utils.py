# utils.py

import config
import boto3
from github import Github
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the GitHub API client
github = Github(config.GITHUB_API_TOKEN)

# Initialize the Bedrock LLM
llm = Bedrock(
    model_id="anthropic.claude-v2",
    region_name=config.AWS_REGION,
    model_kwargs={
        "temperature": 0.7,
        "max_tokens_to_sample": 500
    }
)

# Function to summarize text using a prompt template
def summarize_with_template(text, max_length=170):
    # Load the prompt template
    with open('templates/description_prompt.txt', 'r', encoding='utf-8') as file:
        template_content = file.read()
    # Replace placeholders with actual values
    prompt = template_content.replace('{{ max_length }}', str(max_length))
    prompt = prompt.replace('{{ text }}', text)
    try:
        # Invoke the LLM to get the summary
        response = llm(prompt)
        return response.strip()
    except Exception as e:
        return f"Error during summarization: {str(e)}"

# Function to get recommended projects
def get_recommended_projects(tech_stack, interest_areas):
    query = f"{interest_areas} language:{tech_stack} in:description"
    repositories = github.search_repositories(query=query, sort='stars', order='desc')
    top_repos = []
    for repo in repositories:
        try:
            readme_contents = repo.get_readme().decoded_content.decode('utf-8')
        except Exception:
            readme_contents = "No README available."

        # Get and possibly summarize the description
        description = repo.description or "No description provided."
        if len(description) > 180:
            description = summarize_with_template(description, max_length=170)

        repo_info = {
            'name': repo.full_name,
            'description': description,
            'url': repo.html_url,
            'forks': repo.forks_count,
            'stars': repo.stargazers_count,
            'readme': readme_contents,
        }
        top_repos.append(repo_info)
        if len(top_repos) >= 5:
            break
    return top_repos

def analyze_project_culture(repo_name, readme_contents):
    # Summarize the README content
    summarized_readme = summarize_text(readme_contents)

    # Prepare the prompt
    with open('templates/culture_analysis_prompt.txt', encoding='utf-8') as f:
        prompt_template_content = f.read()
    prompt_template = PromptTemplate(
        input_variables=["repo_name", "readme"],
        template=prompt_template_content,
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    analysis = chain.run(repo_name=repo_name, readme=summarized_readme)
    return analysis.strip()

def generate_contribution_guidelines(repo_name):
    with open('templates/contribution_guidelines_prompt.txt', encoding='utf-8') as f:
        prompt_template_content = f.read()
    prompt_template = PromptTemplate(
        input_variables=["repo_name"],
        template=prompt_template_content,
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    guidelines = chain.run(repo_name=repo_name)
    return guidelines.strip()

def summarize_text(text, max_tokens=500):
    # Limit the text to prevent exceeding context length
    max_chars = 10000  # Adjust as needed
    text = text[:max_chars]

    prompt = f"Please provide a concise summary (max {max_tokens} words) of the following text:\n\n{text}"
    summary = llm(prompt)
    return summary.strip()

# Function to translate text
def translate_text_with_claude(text, target_language):
    prompt = f"Please translate the following text into {target_language}:\n\n{text}"
    try:
        response = llm(prompt)
        return response.strip()
    except Exception as e:
        return f"Error during translation: {str(e)}"
