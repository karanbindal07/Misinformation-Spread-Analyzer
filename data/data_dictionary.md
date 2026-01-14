# Data Dictionary

## Dataset
- data/raw/covid_misinfo_raw.csv

## Unit of observation
- One row represents one tweet (node) from a Twitter conversation graph

## Target variable
- label
  - 1 = misinformation / conspiracy-related
  - 0 = non-conspiracy / factual

## Features

| Column | Description |
|------|------------|
| id | Identifier for the tweet/node |
| time | Relative time index of the tweet within the conversation |
| friends | Number of accounts the user follows |
| followers | Number of followers the user has |
| graph_id | Identifier for the conversation / rumor graph |
| label | Misinformation indicator inferred from graph category |

## Notes
- Labels are inferred from folder structure, not from the original CSV files
- Tweets within the same graph_id are not independent
