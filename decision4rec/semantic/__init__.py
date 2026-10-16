"""Decision models as the semantic layer of the recommender (docs/03_architecture_v2.md, section 9).

The decision model answers only questions whose answer follows from text and general knowledge: what an
item is (facets), how two items relate (relations), and whether a candidate fits a user's constraints. The
trained sequential model keeps the job of learning what people choose.
"""
