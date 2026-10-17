# AM3. Weights of the fused candidate score

Weights of the fused candidate score (Eq. 1): share of the 1,000 validation users of each domain whose next item is among the 30 candidates. Mean: over the three source domains, the basis of the choice. Industrial and Scientific did not enter the choice. The five settings are equivalent on the source domains (the mean recall at 30 differs by at most 0.0016, less than the noise of 1,000 users), so the setting with the largest text weight was kept, because text is what a new scenario has.

| (w_a, w_c, w_t, w_l) | Video Games | Musical Instr. | Steam | mean | Industrial and Sci. |
|---|---|---|---|---|---|
| **Users whose next item is among the 30 candidates (recall@30)** ↑ |  |  |  |  |  |
| (1, 1, 0.3, 0.2) | 17.7% | 9.7% | 18.9% | 15.43% | 8.9% |
| (1, 1.5, 0.3, 0.2) | 17.3% | 9.8% | 19.0% | 15.37% | 8.8% |
| (1.5, 1, 0.3, 0.2) | 17.7% | 9.2% | 19.7% | 15.53% | 8.5% |
| (1, 1, 0.15, 0.1) | 17.2% | 9.7% | 19.4% | 15.43% | 8.6% |
| (1, 1, 0.6, 0.3), kept | 17.8% | 9.9% | 18.5% | 15.40% | 9.3% |
