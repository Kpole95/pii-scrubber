# Hand-labeled OOD set

Store the reviewed real-text examples and annotation guidelines here. Do not commit private data.

## Sampling plan

Target: 200 manually reviewed out-of-distribution examples.

| Category | Target |
|---|---:|
| Names / people | 30 |
| Email addresses | 25 |
| Phone numbers | 25 |
| Addresses / locations | 25 |
| Financial identifiers | 20 |
| Account / ID numbers | 20 |
| Usernames / credentials | 15 |
| Mixed PII | 20 |
| Negative / no PII | 20 |
| **Total** | **200** |

Examples should come from realistic text styles not copied from the
training dataset, including:

- emails and messages
- support conversations
- forms
- notes
- business text
- informal text
- punctuation and formatting variations
- multiple PII entities in one example

Every span must be manually checked against the exact character offsets.