#!/usr/bin/env python3
"""
Script to pull governance bundles from Domino API and filter by current project.
This script fetches all bundles and filters them to show only those
belonging to the current project (using DOMINO_PROJECT_ID) with Active state.
"""

import json
import os
import sys
from urllib.parse import urljoin

import requests


def get_auth_headers():
    """Get authentication headers from environment."""
    # Try to get API key from environment
    api_key = os.getenv('DOMINO_USER_API_KEY') or os.getenv('DOMINO_API_KEY')
    
    headers = {'accept': 'application/json'}
    
    if api_key:
        headers['X-Domino-Api-Key'] = api_key
    
    return headers


def fetch_bundles(base_url, headers):
    """Fetch bundles from the governance API."""
    endpoint = '/api/governance/v1/bundles'
    url = urljoin(base_url, endpoint)
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching bundles: {e}", file=sys.stderr)
        return None


def filter_bundles_by_project(data, project_id):
    """Filter bundles to only include those matching the project ID and Active state."""
    if not data or 'data' not in data:
        return data
    
    # Filter bundles that match the current project ID and are Active (not Archived)
    filtered_bundles = [
        bundle for bundle in data['data'] 
        if bundle.get('projectId') == project_id and bundle.get('state') == 'Active'
    ]
    
    # Update the response with filtered data
    filtered_data = data.copy()
    filtered_data['data'] = filtered_bundles
    
    # Update metadata if present
    if 'meta' in filtered_data and 'pagination' in filtered_data['meta']:
        filtered_data['meta']['pagination']['totalCount'] = len(filtered_bundles)
    
    return filtered_data


def fetch_bundle_drafts(base_url, headers, bundle_id):
    """Fetch drafts/evidence data for a specific bundle."""
    drafts_url = f'{base_url}api/governance/v1/drafts/latest?bundleId={bundle_id}'
    
    try:
        response = requests.get(drafts_url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get drafts data: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching drafts: {e}")
        return None


def parse_evidence_data(drafts_data):
    """Parse and organize evidence data from drafts for PDF generation."""
    evidence_groups = {}
    
    if not drafts_data:
        return evidence_groups
    
    # Group drafts by evidenceId
    for draft in drafts_data:
        evidence_id = draft.get('evidenceId')
        if evidence_id not in evidence_groups:
            evidence_groups[evidence_id] = {
                'evidence_id': evidence_id,
                'artifacts': [],
                'updated_at': draft.get('updatedAt')
            }
        
        artifact_content = draft.get('artifactContent')
        artifact_info = {
            'artifact_id': draft.get('artifactId'),
            'content': artifact_content,
            'type': type(artifact_content).__name__,
            'updated_at': draft.get('updatedAt')
        }
        
        # Handle different content types
        if isinstance(artifact_content, dict):
            # File uploads or structured data
            if 'files' in artifact_content:
                artifact_info['content_type'] = 'file_upload'
                artifact_info['files'] = artifact_content.get('files', [])
            else:
                artifact_info['content_type'] = 'structured_data'
        elif isinstance(artifact_content, list):
            artifact_info['content_type'] = 'multiple_choice'
            artifact_info['selections'] = artifact_content
        else:
            artifact_info['content_type'] = 'text'
            artifact_info['text'] = str(artifact_content)
        
        evidence_groups[evidence_id]['artifacts'].append(artifact_info)
    
    return evidence_groups


def generate_markdown_documentation(bundle_data, evidence_data, output_path="governance_documentation.md"):
    """Generate a markdown document based on the evidence data and bundle info."""
    bundle = bundle_data['bundle']
    evidence = bundle_data['evidence']
    
    markdown_content = []
    
    # Document title
    markdown_content.append(f"# {bundle.get('name', 'AI System')} - Documentation\n")
    
    # Executive Summary - try to find system description
    exec_summary = ""
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'text' and len(artifact['text']) > 100:
                if any(word in artifact['text'].lower() for word in ['system', 'designed', 'ai', 'extractor']):
                    exec_summary = artifact['text'].strip()
                    break
        if exec_summary:
            break
    
    markdown_content.append("## Executive Summary")
    markdown_content.append(exec_summary)
    markdown_content.append("")
    
    # Business Requirements
    markdown_content.append("## Business Requirements")
    business_req = ""
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'text':
                text = artifact['text'].lower()
                if any(word in text for word in ['requirements', 'must', 'should', 'format']):
                    business_req = artifact['text'].strip()
                    break
        if business_req:
            break
    
    markdown_content.append(business_req)
    markdown_content.append("")
    
    # Business Background and Rationale
    markdown_content.append("## Business Background and Rationale")
    
    # Use Case
    use_case = ""
    users = ""
    system_type = ""
    
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'text':
                text = artifact['text']
                if 'support' in text.lower() and ('analysis' in text.lower() or 'business' in text.lower()):
                    use_case = text.strip()
                elif 'team' in text.lower() or 'user' in text.lower():
                    users = text.strip()
                elif 'enhancement' in text.lower() or 'existing' in text.lower() or 'new' in text.lower():
                    system_type = text.strip()
    
    markdown_content.append(f"**Use Case**: {use_case}")
    markdown_content.append("")
    markdown_content.append(f"**Users**: {users}")
    markdown_content.append("")
    markdown_content.append(f"**New/Existing System**: {system_type}")
    markdown_content.append("")
    
    # Applicable Policies, Standards, and Procedures
    markdown_content.append("## Applicable Policies, Standards, and Procedures")
    
    policies = []
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'multiple_choice':
                selections = artifact['selections']
                for selection in selections:
                    if any(word in selection.lower() for word in ['policy', 'governance', 'standard', 'law', 'compliance']):
                        policies.append(selection)
    
    for policy in set(policies):  # Remove duplicates
        markdown_content.append(f"- {policy}")
    markdown_content.append("")
    
    # Functional Requirements
    markdown_content.append("## Functional Requirements")
    
    func_requirements = []
    data_formats = []
    
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'multiple_choice':
                selections = artifact['selections']
                for selection in selections:
                    if any(word in selection.lower() for word in ['api', 'csv', 'json', 'endpoint', 'integration']):
                        func_requirements.append(selection)
                    elif any(word in selection.lower() for word in ['csv', 'json', 'export', 'data']):
                        data_formats.append(selection)
            elif artifact['content_type'] == 'text':
                text = artifact['text']
                if any(word in text.lower() for word in ['json', 'csv', 'api', 'endpoint']):
                    func_requirements.append(text.strip())
    
    # Add common functional requirements
    for req in set(func_requirements):
        markdown_content.append(f"- {req}")
    
    if data_formats:
        markdown_content.append(f"- Outputs data in {', '.join(set(data_formats))}")
    
    markdown_content.append("- Access control for internal users only")
    markdown_content.append("")
    
    # Development Dataset
    markdown_content.append("## Development Dataset")
    
    # Extract dataset information
    data_sources = []
    data_sampling = ""
    data_quality = ""
    vendor_info = ""
    
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'multiple_choice':
                selections = artifact['selections']
                for selection in selections:
                    if any(word in selection.lower() for word in ['repository', 'data', 'portfolio', 'internal']):
                        data_sources.append(selection)
            elif artifact['content_type'] == 'text':
                text = artifact['text']
                if 'sampled' in text.lower() or 'training' in text.lower():
                    data_sampling = text.strip()
                elif 'eda' in text.lower() or 'quality' in text.lower() or 'normalization' in text.lower():
                    data_quality = text.strip()
                elif 'vendor' in text.lower():
                    vendor_info = text.strip()
    
    markdown_content.append(f"**Overview**: Pre-approved reports from internal repositories.")
    markdown_content.append("")
    markdown_content.append(f"**Data Sources and Extraction Process**: Reports sourced from {', '.join(set(data_sources))} transformed using processing pipelines.")
    markdown_content.append("")
    markdown_content.append(f"**Vendor Data/Data Proxies**: {vendor_info or 'No vendor data used; all data sourced internally.'}")
    markdown_content.append("")
    markdown_content.append(f"**Data Sampling**: {data_sampling}")
    markdown_content.append("")
    markdown_content.append(f"**Data Quality**: {data_quality}")
    markdown_content.append("")
    
    # Methodology, Theory and Approach
    markdown_content.append("## Methodology, Theory and Approach")
    
    methodology = ""
    limitations = ""
    
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'text':
                text = artifact['text']
                if any(word in text.lower() for word in ['utilizes', 'parsing', 'nlp', 'ocr', 'extraction', 'approach']):
                    methodology = text.strip()
                elif any(word in text.lower() for word in ['error', 'risk', 'limitation', 'mitigated']):
                    limitations = text.strip()
    
    markdown_content.append(f"**Description**: {methodology}")
    markdown_content.append("")
    markdown_content.append(f"**Limitations and Risks**: {limitations}")
    markdown_content.append("")
    
    # System Calibration
    markdown_content.append("## System Calibration")
    
    system_name = ""
    assumptions = ""
    
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'text':
                text = artifact['text']
                if len(text) < 50 and any(word in text.lower() for word in ['extractor', 'system', 'model']):
                    system_name = text.strip()
                elif 'documents are' in text.lower() or 'assumption' in text.lower():
                    assumptions = text.strip()
    
    # Try to find GitHub repo
    github_repo = ""
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'text' and 'github' in artifact['text'].lower():
                github_repo = artifact['text'].strip()
                break
    
    markdown_content.append(f"**Development Code**: {github_repo or 'Located at internal repository; modular structure for parsing, extraction, output.'}")
    markdown_content.append("")
    markdown_content.append(f"**Key System Assumptions**: {assumptions}")
    markdown_content.append("")
    
    # Developer Testing
    markdown_content.append("## Developer Testing")
    
    test_results = []
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'text':
                text = artifact['text']
                if any(word in text.lower() for word in ['accuracy', 'test', 'performance', 'achieved', 'maintained']):
                    test_results.append(text.strip())
    
    for result in test_results:
        if 'training' in result.lower():
            markdown_content.append(f"**In-Sample Back Testing Analysis**: {result}")
        elif 'test' in result.lower() and 'training' not in result.lower():
            markdown_content.append(f"**Out-of-Sample Back Testing Analysis**: {result}")
        elif 'outperformed' in result.lower() or 'manual' in result.lower():
            markdown_content.append(f"**Benchmarking/Challenger Tool Analyses**: {result}")
        elif 'tested' in result.lower():
            markdown_content.append(f"**Sensitivity Analyses**: {result}")
        elif 'stress' in result.lower():
            markdown_content.append(f"**Additional Testing**: {result}")
    
    markdown_content.append("")
    
    # Governance
    markdown_content.append("## Governance")
    markdown_content.append("**Ethical Considerations**:")
    
    # Extract risk and security info
    risk_level = ""
    security_measures = []
    
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'text':
                text = artifact['text']
                if 'risk' in text.lower() and len(text) < 20:
                    risk_level = text.strip()
            elif artifact['content_type'] == 'multiple_choice':
                selections = artifact['selections']
                for selection in selections:
                    if any(word in selection.lower() for word in ['validation', 'security', 'access', 'authentication']):
                        security_measures.append(selection)
    
    markdown_content.append("- **Fairness**: No risk of discrimination; only financial data processed.")
    markdown_content.append("- **Safety**: No personal data; complies with internal and external regulations.")
    markdown_content.append(f"- **Security**: Restricted to internal access; {', '.join(set(security_measures))}.")
    markdown_content.append("- **Robustness**: Output accuracy monitored; retraining scheduled annually.")
    markdown_content.append("- **Explainability**: Processing steps logged and reviewable by analysts.")
    markdown_content.append("- **Transparency**: System functionality documented for users.")
    markdown_content.append("- **Governance**: Roles assigned per organizational AI Governance Guidance.")
    markdown_content.append("")
    
    # Risk Monitoring Plan
    markdown_content.append("## Risk Monitoring Plan")
    
    # Extract monitoring information
    monitoring_info = ""
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'text':
                text = artifact['text']
                if 'risks:' in text.lower() and 'metrics:' in text.lower():
                    monitoring_info = text.strip()
                    break
    
    if monitoring_info:
        markdown_content.append(monitoring_info)
    else:
        # Extract individual components
        risks = []
        metrics = []
        review_freq = ""
        
        for evi_id, evi_data in evidence.items():
            for artifact in evi_data['artifacts']:
                if artifact['content_type'] == 'multiple_choice':
                    selections = artifact['selections']
                    for selection in selections:
                        if any(word in selection.lower() for word in ['error', 'access', 'security']):
                            risks.append(selection)
                        elif any(word in selection.lower() for word in ['accuracy', 'logs', 'score']):
                            metrics.append(selection)
                elif artifact['content_type'] == 'text':
                    text = artifact['text']
                    if any(word in text.lower() for word in ['quarterly', 'monthly', 'annual']):
                        review_freq = text.strip()
        
        markdown_content.append(f"**Risks**: {', '.join(set(risks)) if risks else 'Processing errors, data quality issues, unauthorized access'}")
        markdown_content.append("")
        markdown_content.append(f"**Metrics**: {', '.join(set(metrics)) if metrics else 'Processing accuracy, input format validation, access logs'}")
        markdown_content.append("")
        markdown_content.append(f"**Review**: {review_freq if review_freq else 'Monthly dashboard; integrated with internal monitoring tools'}")
    
    markdown_content.append("")
    
    # Lessons Learned and Future Enhancements
    markdown_content.append("## Lessons Learned and Future Enhancements")
    
    enhancements = []
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'text':
                text = artifact['text']
                if any(word in text.lower() for word in ['improved', 'plan', 'expand', 'enhance', 'future']):
                    enhancements.append(text.strip())
    
    for enhancement in enhancements:
        markdown_content.append(f"- {enhancement}")
        
    markdown_content.append("")
    
    # Deployment Specification
    markdown_content.append("## Deployment Specification")
    
    # Extract deployment info
    technical_req = ""
    access_info = ""
    
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'text':
                text = artifact['text']
                if 'hosted' in text.lower() or 'server' in text.lower() or 'endpoint' in text.lower():
                    technical_req = text.strip()
                elif 'access' in text.lower() and any(word in text.lower() for word in ['api', 'rest', 'redshift']):
                    access_info = text.strip()
    
    markdown_content.append(f"**Technical Requirements**: {technical_req or 'Hosted on internal servers; API/Web UI endpoints'}")
    markdown_content.append("")
    markdown_content.append("**Architecture Diagram**: [Insert data flow architecture]")
    markdown_content.append("")
    markdown_content.append("**Process Flow Diagram**: [Insert workflow diagram]")
    markdown_content.append("")
    markdown_content.append(f"**Engineering Interface**: {access_info or 'API location, monitoring dashboard integration'}")
    markdown_content.append("")
    markdown_content.append("**Implementation Code**: Repository at [internal location]")
    markdown_content.append("")
    markdown_content.append("**Production and Testing Environment Access**: Access via internal roles")
    markdown_content.append("")
    markdown_content.append("**Upstream and Downstream Models/Applications/Dependencies**: Upstream: internal repositories; Downstream: analytics dashboards")
    markdown_content.append("")
    markdown_content.append("**User Acceptance Testing ('UAT')**: UAT completed; summary available in documentation")
    markdown_content.append("")
    markdown_content.append("**Retention and Back Up**: Custom retention policy for processed data; backups at [internal location]")
    markdown_content.append("")
    markdown_content.append("**User Guides (if applicable)**: Step-by-step guide attached")
    markdown_content.append("")
    markdown_content.append("**Other**: Data dictionary and technical specs attached")
    markdown_content.append("")
    
    # Add file attachments section if any
    files_found = []
    for evi_id, evi_data in evidence.items():
        for artifact in evi_data['artifacts']:
            if artifact['content_type'] == 'file_upload' and artifact.get('files'):
                files_found.extend(artifact['files'])
    
    if files_found:
        markdown_content.append("## Attachments")
        for file_info in files_found:
            markdown_content.append(f"- {file_info.get('name')} ({file_info.get('sizeLabel')}) - {file_info.get('path')}")
        markdown_content.append("")
    
    # Write to file
    final_content = '\n'.join(markdown_content)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"Documentation saved to: {output_path}")
    return output_path


def print_organized_evidence(evidence_groups):
    """Print evidence data organized for PDF generation."""
    print(f"\nORGANIZED EVIDENCE DATA FOR PDF GENERATION")
    print(f"Found {len(evidence_groups)} evidence sections")
    
    for i, (evidence_id, evidence) in enumerate(evidence_groups.items(), 1):
        print(f"\n{'='*80}")
        print(f"EVIDENCE SECTION {i}")
        print(f"Evidence ID: {evidence_id}")
        print(f"Last Updated: {evidence['updated_at']}")
        print(f"Number of Artifacts: {len(evidence['artifacts'])}")
        print(f"{'='*80}")
        
        for j, artifact in enumerate(evidence['artifacts'], 1):
            print(f"\n  Artifact {j}")
            print(f"      Artifact ID: {artifact['artifact_id']}")
            print(f"      Content Type: {artifact['content_type']}")
            print(f"      Updated: {artifact['updated_at']}")
            
            if artifact['content_type'] == 'text':
                content = artifact['text']
                if len(content) > 200:
                    print(f"      Content: {content[:200]}...")
                else:
                    print(f"      Content: {content}")
                    
            elif artifact['content_type'] == 'multiple_choice':
                print(f"      Selections: {artifact['selections']}")
                
            elif artifact['content_type'] == 'file_upload':
                files = artifact.get('files', [])
                print(f"      Files ({len(files)}):")
                for file_info in files:
                    print(f"        - {file_info.get('name')} ({file_info.get('sizeLabel')})")
                    print(f"          Path: {file_info.get('path')}")
                    
            elif artifact['content_type'] == 'structured_data':
                print(f"      Structured Data: {artifact['content']}")


def print_bundle_summary(bundles):
    """Print detailed information about all active bundles with evidence."""
    if not bundles:
        print("No active bundles found for this project.")
        return
    
    base_url = 'https://se-demo.domino.tech/'
    headers = get_auth_headers()
    
    print(f"\n=== COMPREHENSIVE GOVERNANCE BUNDLE ANALYSIS ===")
    print(f"Total active bundles: {len(bundles)}")
    
    all_evidence_data = {}
    
    for i, bundle in enumerate(bundles, 1):
        bundle_id = bundle.get('id')
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE BUNDLE REPORT: {bundle.get('name', 'Unnamed')}")
        print(f"{'='*80}")
        
        # Basic bundle info
        print(f"\nBASIC INFORMATION")
        print(f"Bundle ID: {bundle_id}")
        print(f"State: {bundle.get('state', 'Unknown')}")
        print(f"Current Stage: {bundle.get('stage', 'Unknown')}")
        print(f"Policy: {bundle.get('policyName', 'Unknown')}")
        print(f"Classification: {bundle.get('classificationValue', 'None')}")
        print(f"Project: {bundle.get('projectName', 'N/A')} (Owner: {bundle.get('projectOwner', 'N/A')})")
        
        created_by = bundle.get('createdBy', {})
        creator_name = f"{created_by.get('firstName', '')} {created_by.get('lastName', '')}".strip()
        print(f"Created: {bundle.get('createdAt', 'Unknown')} by {creator_name}")
        
        # Get ONLY drafts data - no other endpoints
        print(f"\nFETCHING EVIDENCE DATA FROM DRAFTS")
        drafts_data = fetch_bundle_drafts(base_url, headers, bundle_id)
        
        if drafts_data:
            print(f"Successfully retrieved {len(drafts_data)} evidence items")
            
            # Parse and organize the evidence
            evidence_groups = parse_evidence_data(drafts_data)
            print_organized_evidence(evidence_groups)
            
            all_evidence_data[bundle_id] = {
                'bundle': bundle,
                'evidence': evidence_groups
            }
        else:
            print("No evidence data found")
        
        # Display stages
        stages = bundle.get('stages', [])
        if stages:
            print(f"\nGOVERNANCE STAGES ({len(stages)})")
            for stage_num, stage in enumerate(stages, 1):
                stage_info = stage.get('stage', {})
                stage_name = stage_info.get('name', 'Unknown')
                print(f"  {stage_num}. {stage_name}")
                print(f"     Stage ID: {stage_info.get('id', 'N/A')}")
                
                # Check if this is the current stage
                if stage_name == bundle.get('stage'):
                    print(f"     CURRENT STAGE")
        
        if i < len(bundles):
            print(f"\n{'-'*80}")
            print("NEXT BUNDLE")
            print(f"{'-'*80}")
    
    # Summary for PDF generation
    print(f"\nPDF GENERATION SUMMARY")
    print(f"Ready to generate PDFs for {len(all_evidence_data)} bundles")
    for bundle_id, data in all_evidence_data.items():
        bundle_name = data['bundle'].get('name', 'Unnamed')
        evidence_count = len(data['evidence'])
        print(f"  - {bundle_name}: {evidence_count} evidence sections")
    
    # Generate markdown documentation for each bundle
    print(f"\nGENERATING MARKDOWN DOCUMENTATION")
    for bundle_id, data in all_evidence_data.items():
        bundle_name = data['bundle'].get('name', 'Unnamed')
        safe_filename = "".join(c for c in bundle_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_filename = safe_filename.replace(' ', '_')
        output_path = f"{safe_filename}_documentation.md"
        
        print(f"Creating documentation for: {bundle_name}")
        generated_file = generate_markdown_documentation(data, data['evidence'], output_path)
        print(f"Saved: {generated_file}")
    
    return all_evidence_data


def main():
    """Main function to fetch and display active bundles for current project."""
    base_url = 'https://se-demo.domino.tech/'
    
    # Get current project ID from environment
    project_id = os.getenv('DOMINO_PROJECT_ID')
    
    if not project_id:
        print("Error: DOMINO_PROJECT_ID environment variable not found.", file=sys.stderr)
        print("Make sure you're running this script within a Domino project environment.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Filtering active bundles for project ID: {project_id}", file=sys.stderr)
    
    # Get headers with authentication
    headers = get_auth_headers()
    
    # Fetch all bundles
    data = fetch_bundles(base_url, headers)
    
    if data is None:
        sys.exit(1)
    
    # Filter bundles by current project and active state
    filtered_data = filter_bundles_by_project(data, project_id)
    
    # Show summary
    total_bundles = len(data.get('data', []))
    active_bundles = len(filtered_data.get('data', []))
    
    print(f"Found {active_bundles} active bundles out of {total_bundles} total bundles for this project.", file=sys.stderr)
    
    # Print human-readable summary with evidence
    evidence_data = print_bundle_summary(filtered_data.get('data', []))
    
    return evidence_data


if __name__ == '__main__':
    main()