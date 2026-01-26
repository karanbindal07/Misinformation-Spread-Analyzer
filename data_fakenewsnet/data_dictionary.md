# Data Dictionary — FakeNewsNet (PolitiFact CSV subset)

## Raw Files
- `data_fakenewsnet/raw/politifact/politifact_real.csv`
- `data_fakenewsnet/raw/politifact/politifact_fake.csv`

## Unit of Observation
- One PolitiFact fact checked news item (one row per item)

## Target Variable
- `label`
  - 0 = real
  - 1 = fake

## Expected Core Fields (varies by release)
Typical columns in FakeNewsNet CSV exports may include:
- `id` or `news_id`: unique identifier for the news item
- `title`: headline/title text
- `text` / `content` / `news_text`: article text (if included)
- `url`: original article URL
- `publish_date`: publication date (if included)
- other source-related metadata (publisher/domain, etc.)

## Notes
- Exact column names can differ depending on how the CSV was generated.
- In Stage 3B, we will run a schema check and document the exact columns we observe.
- Raw data is ignored by git; processed dataset(s) will be created under `data_fakenewsnet/processed/`.
