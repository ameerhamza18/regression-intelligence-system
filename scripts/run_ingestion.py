from regression_intelligence.data.ingestion import ingest

if __name__ == "__main__":
    df = ingest()
    print(df.head())
    print(df.dtypes)
