import pandas as pd
import numpy as np
import os
from datetime import datetime
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset


# ── Config ────────────────────────────────────────────────
REFERENCE_DATA_PATH = "data/processed/X_train.csv"
REPORTS_DIR         = "monitoring/reports"
DRIFT_THRESHOLD     = 0.5

KEY_FEATURES = [
    'TransactionAmt_log',
    'hour',
    'has_identity',
    'V257',
    'V246',
    'V244',
    'V242',
    'V201',
    'ProductCD',
    'card4'
]


def load_reference_data(sample_size: int = 5000) -> pd.DataFrame:
    df = pd.read_csv(REFERENCE_DATA_PATH)
    return df[KEY_FEATURES].sample(n=sample_size, random_state=42)


def simulate_current_data(
    reference: pd.DataFrame,
    drift_scenario: str = "normal"
) -> pd.DataFrame:
    current = reference.copy()

    if drift_scenario == "normal":
        num_cols = current.select_dtypes(include=[np.number]).columns
        noise = np.random.normal(0, 0.01, current[num_cols].shape)
        current[num_cols] = current[num_cols] + noise

    elif drift_scenario == "moderate":
        current['TransactionAmt_log'] = current['TransactionAmt_log'] * 1.3
        current['hour'] = (current['hour'] + 4) % 24
        current['has_identity'] = np.random.choice(
            [0, 1], size=len(current), p=[0.6, 0.4]
        )

    elif drift_scenario == "severe":
        current['TransactionAmt_log'] = current['TransactionAmt_log'] * 2.0
        current['hour'] = np.random.randint(0, 8, size=len(current))
        current['has_identity'] = 1
        current['V257'] = current['V257'] * 0.1
        current['V246'] = current['V246'] * 0.1
        current['ProductCD'] = np.random.randint(0, 5, size=len(current))

    return current


def extract_drift_results(result_dict: dict) -> tuple:
    """
    Extract drifted column count and share from Evidently 0.7.x output.
    Returns (drifted_count, drift_share)
    """
    drifted_count = 0
    drift_share   = 0.0

    metric_results = result_dict.get('metric_results', {})
    for key, value in metric_results.items():
        if isinstance(value, dict):
            # Extract count of drifted columns
            count_data = value.get('count', {})
            if isinstance(count_data, dict) and 'value' in count_data:
                drifted_count = int(count_data['value'])

            # Extract share of drifted columns
            share_data = value.get('share', {})
            if isinstance(share_data, dict) and 'value' in share_data:
                drift_share = round(float(share_data['value']), 4)

    return drifted_count, drift_share


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    scenario: str
) -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Wrap in Evidently Dataset
    ref_dataset = Dataset.from_pandas(
        reference,
        data_definition=DataDefinition()
    )
    cur_dataset = Dataset.from_pandas(
        current,
        data_definition=DataDefinition()
    )

    # Build and run report
    report   = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=ref_dataset,
        current_data=cur_dataset
    )

    # Save HTML report
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{REPORTS_DIR}/drift_report_{scenario}_{timestamp}.html"
    snapshot.save_html(report_path)

    # Extract results
    result_dict              = snapshot.dump_dict()
    drifted_count, drift_share = extract_drift_results(result_dict)
    drift_flag               = drift_share > DRIFT_THRESHOLD

    summary = {
        "scenario"          : scenario,
        "timestamp"         : timestamp,
        "features_monitored": len(KEY_FEATURES),
        "drifted_features"  : drifted_count,
        "drift_score"       : drift_share,
        "drift_detected"    : drift_flag,
        "alert"             : "RETRAINING REQUIRED" if drift_flag else "NO ACTION NEEDED",
        "report_path"       : report_path
    }

    return summary


def print_summary(summary: dict):
    print("=" * 60)
    print(f"DRIFT DETECTION REPORT — {summary['scenario'].upper()}")
    print("=" * 60)
    print(f"Timestamp           : {summary['timestamp']}")
    print(f"Features Monitored  : {summary['features_monitored']}")
    print(f"Drifted Features    : {summary['drifted_features']}")
    print(f"Drift Score         : {summary['drift_score']} "
          f"(threshold: {DRIFT_THRESHOLD})")
    print(f"Drift Detected      : {summary['drift_detected']}")
    print(f"Alert               : {summary['alert']}")
    print(f"Report Saved        : {summary['report_path']}")
    print("=" * 60)


if __name__ == "__main__":
    print("Loading reference data...")
    reference = load_reference_data(sample_size=5000)
    current_size = 1000

    for scenario in ["normal", "moderate", "severe"]:
        print(f"\nRunning scenario: {scenario}")
        current = simulate_current_data(
            reference.sample(n=current_size, random_state=42),
            drift_scenario=scenario
        )
        summary = run_drift_report(reference, current, scenario)
        print_summary(summary)