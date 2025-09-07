---
# Styling knobs for Pandoc/typst/HTML exporters
fontsize: 11pt
linestretch: 1.15
geometry: margin=1in
numbersections: true
toc: true
toc-depth: 2
lang: en-US
header-includes:
  - |
    <style>
      html, body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
        color: #111;
        line-height: 1.55;
      }
      h1, h2, h3 { font-weight: 700; line-height: 1.25; margin: 0 0 .35rem 0; }
      h1 { font-size: 1.9rem; letter-spacing: .2px; margin-top: 0; }
      h2 { font-size: 1.3rem; margin-top: 1.4rem; border-bottom: 1px solid #e5e7eb; padding-bottom: .25rem; }
      h3 { font-size: 1.05rem; margin-top: 1.0rem; }
      p  { margin: .35rem 0 .55rem; }
      ul, ol { margin: .35rem 0 .55rem .9rem; }
      dl { margin: .35rem 0 .55rem; }
      dt { font-weight: 600; }
      dd { margin: 0 0 .45rem 1.0rem; }
      code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, 'Cascadia Mono', Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: .92em; }
      pre { background:#fbfbfb; border:1px solid #eee; border-radius:6px; padding:.6rem .8rem; }
      blockquote { color:#374151; border-left: 3px solid #e5e7eb; margin:.6rem 0; padding:.25rem .8rem; }
      hr { border:none; border-top:1px solid #e5e7eb; margin: 1.0rem 0; }
      .meta small { color:#374151; }
      .doc-title { font-variant-caps: all-small-caps; letter-spacing:.5px; color:#0f172a; }
    </style>
---

# <span class="doc-title">HelpBot v2.3 [Internal Policy Update] – Documentation</span>

<div class="meta">
<small><strong>Policy:</strong> Conditional v0.17 &nbsp; • &nbsp; <strong>Stage:</strong> AI System Documentation</small><br>
<small><strong>Project:</strong> arize-demo-integration (Owner: nick_goble) &nbsp; • &nbsp; <strong>Created:</strong> September 07, 2025</small>
</div>

---


## Executive Summary

Reports sampled by year and sector for training/testing.

Performed EDA to identify inconsistencies, missing fields; applied imputation and normalization.

no-vendor

finrep-extractor

internal

finrep-restricted

finrep-methodology

enhancement

Internal Fitch analytics teams and business units requiring financial data for analysis.

document-sector

github.com/fitch-field/helpbotmodel

All documents are genuine, pre-approved financial reports; no personal data present.

Tested on reports with missing headers, scanned images; robust except for very poor scans.

high

quarterly

monthly-annual

Outputs data in JSON/CSV

Plan to expand to additional report types and automate error correction

- Internal Fitch repositories

- Historical portfolio data

- Public press releases

- Portfolio and market data

- CSV data exports

- Summarized text content

- Input validation to prevent manipulation

- Extraction/processing errors

- Unauthorized access

- Fitch Data Governance Policy

- Copyright laws


<hr style="border:none;border-top:1px solid #ddd;margin:1.0rem 0;"/>


## Business Requirements

_No entries_


## Business Background and Rationale

_No entries_


## Applicable Policies, Standards, and Procedures

_No entries_


## Functional Requirements

_No entries_


## Development Dataset

_No entries_


## Methodology, Theory and Approach

Utilizes document parsing, OCR, and NLP for metric extraction. Based on information extraction theory and deep learning approaches

Improved OCR preprocessing for scanned documents


<hr style="border:none;border-top:1px solid #ddd;margin:1.0rem 0;"/>


## System Calibration

_No entries_


## Developer Testing

The FINREP Extractor is a custom-built AI system designed for the automated extraction of key financial metrics and data points from uploaded financial reports. It is embedded within Fitch’s internal suite of analytics tools, accessible via API and Web UI for internal users. Its primary function is to streamline data acquisition for internal analysis, improving efficiency and accuracy in financial reporting workflows.

Stress tested with batch uploads and large files.

Achieved \>95% extraction accuracy on training set.

Maintained \>92% accuracy on test set.

Outperformed manual extraction in speed and accuracy.

Risks: Extraction errors, data quality issues, unauthorized access
Metrics: Extraction accuracy, input format validation, access logs
Review: Monthly dashboard; integrated with internal monitoring tools

- Summary accuracy (ROUGE scores)

- Access logs and security events


<hr style="border:none;border-top:1px solid #ddd;margin:1.0rem 0;"/>


## Governance

_No entries_


## Risk Monitoring Plan

Possible extraction errors from poorly formatted or scanned documents; mitigated with pre-processing, but residual risk remains.

The system supports internal financial analysis by automating the extraction of key metrics from financial reports, reducing manual data entry and potential errors.

low-risk


<hr style="border:none;border-top:1px solid #ddd;margin:1.0rem 0;"/>


## Lessons Learned and Future Enhancements

_No entries_


## Deployment Specification

AWS Redshift Access, Atlas Access via REST API

Hosted on internal servers; API/Web UI endpoints

- RESTful API endpoints

- Authentication systems

- API integration capabilities

**Attachments**

- **images.png** 8.8 K
  
  Path:
  ```
  govern-metadata/8c1e4559-549c-42e8-a05a-760c99665349/b3d07aa3-076b-4313-971d-4ccfea2c27ab/images.png
  ```


<hr style="border:none;border-top:1px solid #ddd;margin:1.0rem 0;"/>
