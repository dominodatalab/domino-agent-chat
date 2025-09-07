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


def print_bundle_summary(bundles):
    """Print a summary of the bundles."""
    if not bundles:
        print("No active bundles found for this project.")
        return
    
    print(f"\n=== GOVERNANCE BUNDLES SUMMARY ===")
    print(f"Total active bundles: {len(bundles)}")
    
    # Group by state
    states = {}
    for bundle in bundles:
        state = bundle.get('state', 'Unknown')
        states[state] = states.get(state, 0) + 1
    
    print(f"\nBy State:")
    for state, count in states.items():
        print(f"  {state}: {count}")
    
    # Group by policy
    policies = {}
    for bundle in bundles:
        policy = bundle.get('policyName', 'Unknown')
        policies[policy] = policies.get(policy, 0) + 1
    
    print(f"\nBy Policy:")
    for policy, count in policies.items():
        print(f"  {policy}: {count}")
    
    # Show individual bundles
    print(f"\n=== BUNDLE DETAILS ===")
    for bundle in bundles:
        print(f"\nBundle: {bundle.get('name', 'Unnamed')}")
        print(f"  ID: {bundle.get('id', 'N/A')}")
        print(f"  State: {bundle.get('state', 'Unknown')}")
        print(f"  Stage: {bundle.get('stage', 'Unknown')}")
        print(f"  Policy: {bundle.get('policyName', 'Unknown')}")
        print(f"  Created: {bundle.get('createdAt', 'Unknown')}")
        print(f"  Created by: {bundle.get('createdBy', {}).get('firstName', '')} {bundle.get('createdBy', {}).get('lastName', '')}")
        
        # Show attachments if present
        attachments = bundle.get('attachments', [])
        if attachments:
            print(f"  Attachments ({len(attachments)}):")
            for att in attachments:
                att_type = att.get('type', 'Unknown')
                if att_type == 'ModelVersion':
                    name = att.get('identifier', {}).get('name', 'Unknown')
                    version = att.get('identifier', {}).get('version', 'Unknown')
                    print(f"    - Model: {name} v{version}")
                else:
                    print(f"    - {att_type}")


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
    
    # Print human-readable summary
    print_bundle_summary(filtered_data.get('data', []))


if __name__ == '__main__':
    main()