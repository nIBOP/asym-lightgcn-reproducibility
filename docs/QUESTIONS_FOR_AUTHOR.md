# Questions for Final Public Release

These choices require author confirmation before publication.

1. Repository owner/account:
   - Confirmed: `nIBOP`.

2. Repository name:
   - Confirmed: `asym-lightgcn-reproducibility`.

3. License:
   - Confirmed: MIT for code.
   - Dataset terms remain governed by the original dataset providers.

4. Persistent identifier:
   - Proposed: create GitHub repository first, then connect it to Zenodo and mint DOI.
   - If time is short, submit with GitHub URL first and add Zenodo later if journal requests DOI.

5. Checkpoints:
   - Proposed: do not publish checkpoints in GitHub.
   - Reason: large size, derived from source datasets, and not necessary for reproducing paper-level results if scripts and aggregate logs are available.

6. Prepared split/data files:
   - Proposed: do not publish `.inter`/`.item` files.
   - Instead: publish deterministic preparation instructions, split protocol, seeds, and aggregate split statistics.
