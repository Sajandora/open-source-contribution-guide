# app.py

import streamlit as st
import os
import logging
import time
from jinja2 import Template
from utils import (
    get_recommended_projects,
    analyze_project_culture,
    generate_contribution_guidelines,
    summarize_text,
    translate_text_with_claude,
)
import config
import boto3
import pdfkit

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize session state
if 'analyzed_projects' not in st.session_state:
    st.session_state['analyzed_projects'] = {}
if 'recommended_projects' not in st.session_state:
    st.session_state['recommended_projects'] = []
if 'search_performed' not in st.session_state:
    st.session_state['search_performed'] = False

# AWS S3 Configuration
S3_BUCKET_NAME = config.S3_BUCKET_NAME
S3_REGION_NAME = config.AWS_REGION
s3_client = boto3.client('s3', region_name=S3_REGION_NAME)

# Streamlit App Configuration
st.set_page_config(page_title="Open Source Contribution Guide", layout="wide")
st.title("Open Source Contribution Guide")

def format_number(num):
    if num >= 1000000:
        return f"{num / 1000000:.1f}M"
    elif num >= 1000:
        return f"{num / 1000:.1f}k"
    else:
        return str(num)

def analyze_project(idx):
    project = st.session_state['recommended_projects'][idx]
    with st.spinner(f"Analyzing culture for {project['name']}..."):
        culture_analysis = analyze_project_culture(project['name'], project['readme'])
        st.session_state['analyzed_projects'][idx]['culture_analysis'] = culture_analysis

    with st.spinner(f"Generating guidelines for {project['name']}..."):
        guidelines = generate_contribution_guidelines(project['name'])
        st.session_state['analyzed_projects'][idx]['guidelines'] = guidelines

    # Force rerun to update the UI
    st.experimental_rerun()

# 1. User Input Stage
st.header("1. User Input")

with st.form(key='user_input_form'):
    tech_stack = st.text_input(
        "Enter your technology stack (e.g., Python, JavaScript):", value=""
    )
    interest_areas = st.text_input(
        "Enter your areas of interest (e.g., web development, data science):", value=""
    )
    available_hours = st.number_input(
        "Enter the number of hours you can contribute per week:",
        min_value=1,
        max_value=40,
        value=5,
    )
    target_language = st.text_input("Enter the target language for summaries (optional):", value="")
    submit_button = st.form_submit_button(label='Find Projects')

# Update search_performed state when form is submitted
if submit_button:
    if not tech_stack.strip() or not interest_areas.strip():
        st.error("Please provide both your technology stack and areas of interest.")
    else:
        st.session_state['search_performed'] = True
        with st.spinner("Fetching recommended projects..."):
            recommended_projects = get_recommended_projects(tech_stack, interest_areas)
            st.session_state['recommended_projects'] = recommended_projects
            # Initialize analyzed_projects entries for new search
            st.session_state['analyzed_projects'] = {
                idx: {
                    'project_info': project,
                    'culture_analysis': None,
                    'guidelines': None
                }
                for idx, project in enumerate(recommended_projects)
            }

# Display projects if search has been performed
if st.session_state['search_performed']:
    st.header("2. Project Recommendations")

    if not st.session_state['recommended_projects']:
        st.warning("No projects found. Please try different inputs.")
    else:
        for idx, project in enumerate(st.session_state['recommended_projects']):
            st.subheader(f"{idx + 1}. {project['name']}")
            st.write(f"**Description:** {project['description']}")

            # Format stars and forks
            formatted_stars = format_number(project['stars'])
            formatted_forks = format_number(project['forks'])
            st.write(f"**Stars:** {formatted_stars} ⭐")
            st.write(f"**Forks:** {formatted_forks} 🍴")
            st.write(f"**URL:** [{project['url']}]({project['url']})")

            with st.spinner(f"Generating summary for {project['name']}..."):
                summary = summarize_text(project['readme'])

            # Translation option
            if target_language.strip():
                translated_summary = translate_text_with_claude(summary, target_language)
                st.markdown(f"**Summary ({target_language}):**")
                st.write(translated_summary)
            else:
                st.markdown("**Summary:**")
                st.write(summary)

            # Create unique key for analyze button
            analyze_key = f"analyze_button_{idx}"

            # Check if project has been analyzed
            project_data = st.session_state['analyzed_projects'].get(idx, {})
            if project_data.get('culture_analysis') is None:
                if st.button(f"Analyze {project['name']}", key=analyze_key):
                    analyze_project(idx)
            else:
                # Display existing analysis
                st.markdown("### Culture Analysis")
                st.write(project_data['culture_analysis'])
                st.markdown("### Contribution Guidelines")
                st.write(project_data['guidelines'])

            st.markdown("---")

        # PDF Generation Section
        if any(data.get('culture_analysis') is not None for data in st.session_state['analyzed_projects'].values()):
            if st.button("Generate PDF and Upload to S3"):
                with st.spinner("Generating PDF and uploading to S3..."):
                    try:
                        # Collect only analyzed projects for PDF
                        project_data = [
                            {
                                'name': data['project_info']['name'],
                                'description': data['project_info']['description'],
                                'url': data['project_info']['url'],
                                'culture_analysis': data['culture_analysis'],
                                'guidelines': data['guidelines']
                            }
                            for data in st.session_state['analyzed_projects'].values()
                            if data['culture_analysis'] is not None
                        ]

                        # Generate HTML content using Jinja2 template
                        template_path = 'templates/pdf_template.html'
                        with open(template_path, encoding='utf-8') as f:
                            template = Template(f.read())

                        html_content = template.render(projects=project_data)

                        # Setup paths
                        output_dir = os.path.join(os.getcwd(), 'output_files')
                        os.makedirs(output_dir, exist_ok=True)
                        html_path = os.path.join(output_dir, 'temp.html')
                        pdf_path = os.path.join(output_dir, 'project_details.pdf')

                        # Generate PDF
                        with open(html_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)

                        wkhtmltopdf_path = '/usr/local/bin/wkhtmltopdf'  # Update if necessary
                        config_pdfkit = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
                        pdfkit.from_file(html_path, pdf_path, configuration=config_pdfkit)

                        # Upload to S3
                        s3_key = f'project_details_{int(time.time())}.pdf'
                        s3_client.upload_file(pdf_path, S3_BUCKET_NAME, s3_key)

                        # Generate download link
                        presigned_url = s3_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_key},
                            ExpiresIn=3600
                        )

                        st.success("PDF generated and uploaded to S3.")
                        st.markdown(f"**Download your PDF here:** [Download PDF]({presigned_url})")

                        # Cleanup
                        os.remove(pdf_path)
                        os.remove(html_path)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        logging.error(f"PDF Generation or S3 Upload Error: {e}")
