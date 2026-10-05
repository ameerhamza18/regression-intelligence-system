import sys

from regression_intelligence.data.ingestion import ingest
from regression_intelligence.data.validation import save_report, validate_dataframe


def main() -> int:
    df = ingest()
    report = validate_dataframe(df)
    path = save_report(report)
    print(report.summary())
    print(f"\nReport saved to {path}")
    return 1 if report.has_critical else 0


if __name__ == "__main__":
    sys.exit(main())
