# Public Release Checklist

Current public repository status:

- [x] Choose final repository owner/account: `nIBOP`.
- [x] Choose final repository name: `asym-lightgcn-reproducibility`.
- [x] Confirm the code license: MIT.
- [x] Set the public GitHub URL in `CITATION.cff`.
- [x] Create the GitHub repository.
- [x] Push the curated package.
- [ ] Confirm no credentials are present before each public release.
- [ ] Confirm no raw data or prepared dataset files are present before each public release.
- [ ] Confirm no `.pt`, `.pth`, `.npy`, `.pkl`, `.inter`, `.item`, `.user` files are present before each public release.
- [ ] Run a file-size audit before each public release.
- [ ] Run a secret scan or at least search for `kaggle`, `token`, `password`, `secret`, `api_key` before each public release.
- [ ] Create a tagged release.
- [ ] Optionally connect the GitHub repository to Zenodo and mint a DOI.
- [ ] Replace the article URL with the final GitHub URL or Zenodo DOI.

Initial publication commands, already completed for the public GitHub repository:

```powershell
git init
git add .
git status
git commit -m "Prepare reproducibility package"
git branch -M main
git remote add origin https://github.com/nIBOP/asym-lightgcn-reproducibility.git
git push -u origin main
```

For a future clean re-publication to a newly created empty repository, run:

```powershell
.\PUBLISH_TO_GITHUB.ps1
```

After creating a Zenodo DOI, update:

- `README.md`;
- `CITATION.cff`;
- the article reproducibility paragraph;
- `paper/REPRODUCIBILITY_MANIFEST_RU.md`.
