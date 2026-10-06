# Local research data

`python -m src.credit download` downloads the official UCI archive into
`data/raw/uci_credit.zip` and verifies the fixed SHA256. Attribution:
Yeh, I. (2009), *Default of Credit Card Clients*, UCI Machine Learning Repository,
[DOI 10.24432/C55S3H](https://doi.org/10.24432/C55S3H), CC BY 4.0.

The source has 30,000 existing-cardholder rows, not rejected loan applications.
Amounts are NT dollars. It has no actual losses, account opening dates, per-customer
snapshot cutoffs or ingestion timestamps. No missing/refused labels are invented.

`data/processed/credit/` holds local split manifests and individual predictions.
The archive, row-level records and model artifacts are ignored by Git; only
aggregate quality, benchmark and diagnostic outputs are published.
The original simulation uses a separate local `data/raw/credit.csv` and must not
be substituted for the real-data benchmark if the source cannot be downloaded.
