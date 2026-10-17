# AM2. How much item content the decision model reads

How much item content the decision model reads. Description: share of the items whose text has a description (on Steam, the tags of the game). Reached: share of the items whose description starts within the first 90 (current) or 250 characters of the item text read by the decision model. Request: tokens of one request with two orders of 30 options, mean over 40 development test users.

|  | Video Games | Musical Instr. | Steam | Industrial and Sci. | MovieLens-1M enriched |
|---|---|---|---|---|---|
| items with a description | 73% | 72% | 100% | 57% | 98% |
| description reached at 90 characters ↑ | 15% | 1% | 98% | 1% | 3% |
| description reached at 250 characters ↑ | 71% | 62% | 100% | 42% | 97% |
| request at 90 characters (tokens) ↓ | 3,549 | 3,632 | 3,882 | 3,924 | 4,239 |
| request at 250 characters (tokens) ↓ | 5,622 | 5,683 | 4,696 | 6,296 | 7,090 |
