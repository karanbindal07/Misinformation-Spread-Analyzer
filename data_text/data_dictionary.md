# Data Dictionary — ISOT Fake News Dataset

## Source
ISOT Fake News Dataset  
Used for educational and research purposes.

## Raw Files
- `True.csv`
- `Fake.csv`

## Processed File
- `isot_text_processed.csv` 

## Columns

| Column | Type | Description |
|------|-----|-------------|
| clean_text | string | Preprocessed article text (lowercased, cleaned, concatenated title + body) |
| label | int | 0 = real news, 1 = fake news |

## Notes
- Dataset focuses on content level misinformation, not spread behavior.