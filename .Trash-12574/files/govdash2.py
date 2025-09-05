#!/usr/bin/env python3
"""
Arize AI Governance Dashboard Generator
Reads config from environment variables (Domino project settings).
"""

from __future__ import annotations
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from arize.exporter import ArizeExportClient
from arize.utils.types import Environments

# -----------------------
# Config (read from env)
# -----------------------

def _get_env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        missing.append(name)
    return val

def _mask(s: str, keep: int = 4) -> str:
    if not s:
        return ""
    return f"{s[:keep]}…{s[-keep:]}" if len(s) > keep * 2 else "****"

missing: list[str] = []

ARIZE_API_KEY       = _get_env("ARIZE_API_KEY", required=True)
ARIZE_SPACE_ID      = _get_env("ARIZE_SPACE_ID", required=True)
ARIZE_PROJECT_NAME  = _get_env("ARIZE_PROJECT_NAME", required=True)  # used as model_id
OPENAI_API_KEY      = _get_env("OPENAI_API_KEY")  # optional here
DAYS_BACK           = int(_get_env("DAYS_BACK", "7"))

if missing:
    # Fail fast with a clean message (no secrets)
    msg = (
        "Missing required environment variables: "
        + ", ".join(missing)
        + "\nSet them in Domino → Project Settings → Environment variables."
    )
    print(msg, file=sys.stderr)
    sys.exit(1)

# -----------------------
# Styling
# -----------------------

FS_COLORS: Dict[str, str] = {
    'primary_blue': '#1B365D',
    'secondary_blue': '#2E5984',
    'accent_blue': '#4A90B8',
    'success_green': '#2E7D32',
    'warning_orange': '#F57C00',
    'danger_red': '#C62828',
    'neutral_gray': '#5F6368',
    'light_gray': '#F8F9FA',
    'white': '#FFFFFF',
    'text_dark': '#212529'
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.color': FS_COLORS['neutral_gray'],
    'grid.linewidth': 0.5,
    'axes.facecolor': FS_COLORS['white'],
    'figure.facecolor': FS_COLORS['white'],
    'text.color': FS_COLORS['text_dark'],
    'axes.labelcolor': FS_COLORS['text_dark'],
    'xtick.color': FS_COLORS['text_dark'],
    'ytick.color': FS_COLORS['text_dark'],
    'axes.edgecolor': FS_COLORS['neutral_gray'],
    'axes.linewidth': 1.0,
    'xtick.major.pad': 6,
    'ytick.major.pad': 6,
    'axes.titlepad': 20,
    'axes.labelpad': 10
})

FS_PALETTE = [
    FS_COLORS['primary_blue'],
    FS_COLORS['secondary_blue'],
    FS_COLORS['accent_blue'],
    FS_COLORS['success_green'],
    FS_COLORS['warning_orange'],
    FS_COLORS['danger_red'],
]
sns.set_palette(FS_PALETTE)

# -----------------------
# Data pull / sample
# -----------------------

def pull_arize_data() -> Optional[pd.DataFrame]:
    print("Connecting to Arize and pulling data...")
    end_time = datetime.now()
    start_time = end_time - timedelta(days=DAYS_BACK)

    try:
        client = ArizeExportClient(api_key=ARIZE_API_KEY)
        df = client.export_model_to_df(
            space_id=ARIZE_SPACE_ID,
            model_id=ARIZE_PROJECT_NAME,
            environment=Environments.TRACING,
            start_time=start_time,
            end_time=end_time,
            columns=[
                'context.span_id',
                'attributes.llm.model_name',
                'attributes.llm.provider',
                'attributes.llm.token_count.total',
                'attributes.llm.token_count.prompt',
                'attributes.llm.token_count.completion',
                'status_code',
                'start_time',
                'end_time',
                'attributes.llm.system',
                'attributes.llm.usage.total_tokens',
                'name'
            ]
        )
        print(f"Pulled {len(df)} records from {start_time.date()} to {end_time.date()}")
        return df
    except Exception as e:
        print(f"Error pulling data from Arize: {e}")
        return None

def load_sample_data() -> pd.DataFrame:
    print("Creating sample data for demo...")
    np.random.seed(42)
    n_records = 100
    end_time = datetime.now()
    start_time = end_time - timedelta(days=DAYS_BACK)
    dates = pd.date_range(start_time, end_time, periods=n_records)

    df = pd.DataFrame({
        'context.span_id': [f'span_{i:04d}' for i in range(n_records)],
        'attributes.llm.model_name': np.random.choice(['gpt-4', 'gpt-3.5-turbo', 'claude-3'], n_records),
        'attributes.llm.provider': np.random.choice(['openai', 'anthropic'], n_records),
        'attributes.llm.token_count.total': np.random.randint(50, 3000, n_records),
        'status_code': np.random.choice(['OK', 'ERROR'], n_records, p=[0.95, 0.05]),
        'start_time': dates,
        'end_time': dates + pd.Timedelta(seconds=np.random.randint(1, 30, n_records)),
        'name': ['ChatCompletion'] * n_records
    })
    print(f"Created {len(df)} sample records")
    return df

# -----------------------
# Cleaning / analysis / viz
# -----------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning data...")
    df = df.copy()
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['hour'] = df['end_time'].dt.hour
    df['date'] = df['end_time'].dt.date
    df['day_of_week'] = df['end_time'].dt.day_name()
    df['duration_seconds'] = (df['end_time'] - df['start_time']).dt.total_seconds()
    df['attributes.llm.token_count.total'] = df['attributes.llm.token_count.total'].fillna(0)
    df['status_code'] = df['status_code'].fillna('UNKNOWN')
    print(f"Data cleaned. Shape: {df.shape}")
    return df

def detect_potential_issues(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if 'attributes.llm.token_count.total' in df.columns:
        q95 = df['attributes.llm.token_count.total'].quantile(0.95)
        n_out = (df['attributes.llm.token_count.total'] > q95).sum()
        if n_out > 0:
            issues.append(f"High token usage detected in {n_out} requests")
    model_std = df.groupby('attributes.llm.model_name')['attributes.llm.token_count.total'].std()
    if len(model_std) and (model_std > model_std.median() * 2).any():
        issues.append("High response variance detected")
    error_rate = (df['status_code'] != 'OK').mean()
    if error_rate > 0.05:
        issues.append(f"Error rate above threshold: {error_rate:.1%}")
    hourly = df.groupby('hour').size()
    if len(hourly) and hourly.max() > hourly.median() * 3:
        issues.append(f"Traffic anomaly detected at hour {hourly.idxmax()}")
    return issues

def create_governance_dashboard(df: pd.DataFrame):
    print("Creating governance dashboard...")
    detected_issues = detect_potential_issues(df)
    fig = plt.figure(figsize=(28, 20))
    fig.patch.set_facecolor(FS_COLORS['white'])

    fig.suptitle('AI GOVERNANCE & RISK MANAGEMENT DASHBOARD',
                 fontsize=28, fontweight='bold', y=0.96,
                 color=FS_COLORS['primary_blue'], x=0.03, ha='left')
    fig.text(0.03, 0.925, 'Real-time Monitoring & Compliance Assessment',
             fontsize=16, color=FS_COLORS['neutral_gray'], style='italic', alpha=0.8)

    # 1
    ax1 = plt.subplot2grid((4, 4), (0, 0), fig=fig)
    model_counts = df['attributes.llm.model_name'].value_counts()
    colors = [FS_COLORS['primary_blue'], FS_COLORS['secondary_blue'], FS_COLORS['accent_blue']][:len(model_counts)]
    wedges, texts, autotexts = plt.pie(model_counts.values, labels=model_counts.index,
                                       autopct='%1.1f%%', colors=colors, startangle=90,
                                       textprops={'fontsize': 10, 'fontweight': '500'})
    for at in autotexts:
        at.set_color('white'); at.set_fontweight('bold')
    ax1.set_title('Model Usage Distribution', fontweight='bold', loc='left',
                  color=FS_COLORS['primary_blue'], pad=25, fontsize=13)

    # 2
    ax2 = plt.subplot2grid((4, 4), (0, 1), fig=fig)
    hourly_tokens = df.groupby('hour')['attributes.llm.token_count.total'].sum()
    plt.plot(hourly_tokens.index, hourly_tokens.values, marker='o', linewidth=3,
             color=FS_COLORS['primary_blue'], markersize=7, markerfacecolor=FS_COLORS['accent_blue'],
             markeredgecolor='white', markeredgewidth=2)
    plt.fill_between(hourly_tokens.index, hourly_tokens.values, alpha=0.15, color=FS_COLORS['primary_blue'])
    ax2.set_title('Token Usage by Hour', fontweight='bold', loc='left',
                  color=FS_COLORS['primary_blue'], pad=25, fontsize=13)
    ax2.set_xlabel('Hour of Day'); ax2.set_ylabel('Total Tokens')

    # 3
    ax3 = plt.subplot2grid((4, 4), (0, 2), fig=fig)
    status_counts = df['status_code'].value_counts()
    colors = [FS_COLORS['success_green'] if 'OK' in str(x) else FS_COLORS['danger_red'] for x in status_counts.index]
    bars = plt.bar(status_counts.index, status_counts.values, color=colors, alpha=0.8,
                   edgecolor='white', linewidth=2)
    for b in bars:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2., h * 1.02, f'{int(h)}',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax3.set_title('System Health Status', fontweight='bold', loc='left',
                  color=FS_COLORS['primary_blue'], pad=25, fontsize=13)
    ax3.set_ylabel('Request Count')

    # 4
    ax4 = plt.subplot2grid((4, 4), (0, 3), fig=fig)
    provider_counts = df['attributes.llm.provider'].value_counts()
    bars = plt.bar(provider_counts.index, provider_counts.values, color=FS_PALETTE[:len(provider_counts)],
                   alpha=0.8, edgecolor='white', linewidth=2)
    for b in bars:
        h = b.get_height()
        ax4.text(b.get_x() + b.get_width()/2., h * 1.02, f'{int(h)}',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax4.set_title('LLM Provider Distribution', fontweight='bold', loc='left',
                  color=FS_COLORS['primary_blue'], pad=25, fontsize=13)
    ax4.set_ylabel('Usage Count')

    # 5
    ax5 = plt.subplot2grid((4, 4), (1, 0), fig=fig)
    vals = df.loc[df['attributes.llm.token_count.total'] > 0, 'attributes.llm.token_count.total']
    plt.hist(vals, bins=20, alpha=0.7, color=FS_COLORS['primary_blue'], edgecolor='white', linewidth=1.5)
    mean_tokens = float(vals.mean()) if len(vals) else 0.0
    plt.axvline(mean_tokens, color=FS_COLORS['danger_red'], linestyle='--', linewidth=3,
                label=f'Mean: {mean_tokens:.0f}', alpha=0.8)
    ax5.set_title('Token Usage Distribution', fontweight='bold', loc='left',
                  color=FS_COLORS['primary_blue'], pad=25, fontsize=13)
    ax5.set_xlabel('Tokens per Request'); ax5.set_ylabel('Frequency'); ax5.legend(loc='upper right')

    # 6
    ax6 = plt.subplot2grid((4, 4), (1, 1), fig=fig)
    pivot = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
    sns.heatmap(pivot, cmap='Blues', cbar_kws={'label': 'Requests', 'shrink': 0.8},
                ax=ax6, linewidths=0.5, linecolor=FS_COLORS['white'])
    ax6.set_title('Usage Pattern Heatmap', fontweight='bold', loc='left',
                  color=FS_COLORS['primary_blue'], pad=25, fontsize=13)
    ax6.set_ylabel('Day of Week')

    # 15 (incident panel)
    ax15 = plt.subplot2grid((4, 4), (3, 2), fig=fig)
    ax15.axis('off')
    incident_text = "INCIDENT DETECTION\n" + "-" * 30 + "\n\n"
    issues = detected_issues
    if issues:
        for it in issues[:3]:
            incident_text += f"- {it}\n\n"
    else:
        incident_text += "No critical issues detected\n\n"
    success_rate = (df['status_code'].str.contains('OK', na=False)).mean() * 100
    status = ("System Health: EXCELLENT" if success_rate >= 95
              else "System Health: GOOD" if success_rate >= 90
              else "System Health: ATTENTION REQUIRED")
    error_count = int((df['status_code'] != 'OK').sum())
    panel = f"{incident_text}GOVERNANCE STATUS\n" + "-" * 30 + f"\n\n{status}\nTotal Errors: {error_count}\n"
    ax15.text(0.05, 0.95, panel, transform=ax15.transAxes, fontsize=10, va='top', fontfamily='monospace',
              bbox=dict(boxstyle="round,pad=0.6", facecolor=FS_COLORS['light_gray'],
                        edgecolor=FS_COLORS['primary_blue'], linewidth=2, alpha=0.9))
    ax15.set_title('Incident Detection & Alerts', fontweight='bold', loc='left',
                   color=FS_COLORS['primary_blue'], pad=25, fontsize=13)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def generate_executive_summary(df: pd.DataFrame):
    print("Generating executive summary...")
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor(FS_COLORS['white'])
    ax.axis('tight'); ax.axis('off')

    total_requests = len(df)
    success_rate = (df['status_code'].str.contains('OK', na=False)).mean() * 100
    total_tokens = df['attributes.llm.token_count.total'].sum()
    avg_tokens = df['attributes.llm.token_count.total'].mean()
    unique_models = df['attributes.llm.model_name'].nunique()
    date_range = f"{df['date'].min()} to {df['date'].max()}"
    peak_hour = df.groupby('hour').size().idxmax()

    rows = [
        ['Total Requests Processed', f'{total_requests:,}', 'INFO'],
        ['System Success Rate', f'{success_rate:.1f}%', 'PASS' if success_rate >= 95 else 'WARN'],
        ['Total Tokens Consumed', f'{int(total_tokens):,}', 'INFO'],
        ['Average Tokens per Request', f'{avg_tokens:.0f}', 'INFO'],
        ['Active Model Count', f'{int(unique_models)}', 'INFO'],
        ['Analysis Period', date_range, 'INFO'],
        ['Peak Usage Hour', f'{peak_hour}:00', 'INFO'],
        ['Data Quality Score', f'{(df["attributes.llm.token_count.total"] > 0).mean() * 100:.1f}%', 'PASS'],
    ]

    table = ax.table(cellText=rows, colLabels=['METRIC', 'VALUE', 'STATUS'],
                     cellLoc='left', loc='center', colWidths=[0.5, 0.3, 0.2])
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1.2, 2.5)

    for c in range(3):
        table[(0, c)].set_facecolor(FS_COLORS['primary_blue'])
        table[(0, c)].set_text_props(weight='bold', color='white', size=14)

    for i in range(1, len(rows)+1):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor(FS_COLORS['light_gray'])
            table[(i, j)].set_text_props(size=11)
            if j == 1:
                table[(i, j)].set_text_props(weight='bold', size=12)

    fig.text(0.02, 0.95, 'AI GOVERNANCE EXECUTIVE SUMMARY',
             fontsize=20, fontweight='bold', color=FS_COLORS['primary_blue'])
    fig.text(0.02, 0.91, 'Key Performance Indicators & Compliance Metrics',
             fontsize=12, color=FS_COLORS['neutral_gray'], style='italic')
    fig.text(0.02, 0.02, f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}',
             fontsize=10, color=FS_COLORS['neutral_gray'])
    return fig

# -----------------------
# Main
# -----------------------

def main():
    print("Starting Arize AI Governance Analysis")
    print("=" * 60)
    print(f"Config → DAYS_BACK={DAYS_BACK}, "
          f"ARIZE_SPACE_ID={_mask(ARIZE_SPACE_ID)}, "
          f"MODEL_ID={ARIZE_PROJECT_NAME}, "
          f"ARIZE_API_KEY={_mask(ARIZE_API_KEY)}")

    df = pull_arize_data()
    if df is None or len(df) == 0:
        print("No real data available. Using sample data for demo.")
        df = load_sample_data()

    df = clean_data(df)
    dashboard_fig = create_governance_dashboard(df)
    summary_fig = generate_executive_summary(df)

    print("Saving plots…")
    dashboard_fig.savefig('ai_governance_dashboard.png', dpi=300, bbox_inches='tight',
                          facecolor='white', edgecolor='none', pad_inches=0.3)
    summary_fig.savefig('governance_executive_summary.png', dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none', pad_inches=0.3)

    print("Analysis complete!")
    print("Enhanced dashboard saved as: ai_governance_dashboard.png")
    print("Enhanced summary saved as: governance_executive_summary.png")

    # Optional: show when running interactively
    # plt.show()

if __name__ == "__main__":
    # In notebooks, IPython adds unknown flags; keep main() arg-free here.
    main()
