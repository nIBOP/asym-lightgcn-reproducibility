# Public Release Checklist

Before creating the public GitHub repository:

- [x] Choose final repository owner/account: `nIBOP`.
- [x] Choose final repository name: `asym-lightgcn-reproducibility`.
- [x] Confirm the code license: MIT.
- [x] Replace placeholder URL in `CITATION.cff`.
- [ ] Confirm no credentials are present.
- [ ] Confirm no raw data or prepared dataset files are present.
- [ ] Confirm no `.pt`, `.pth`, `.npy`, `.pkl`, `.inter`, `.item`, `.user` files are present.
- [ ] Run a file-size audit.
- [ ] Run a secret scan or at least search for `kaggle`, `token`, `password`, `secret`, `api_key`.
- [ ] Create the GitHub repository.
- [ ] Push the curated package.
- [ ] Create a tagged release.
- [ ] Optionally connect the GitHub repository to Zenodo and mint a DOI.
- [ ] Replace the article URL with the final GitHub URL or Zenodo DOI.

Suggested commands from this package directory:

```powershell
git init
git add .
git status
git commit -m "Prepare reproducibility package"
git branch -M main
git remote add origin https://github.com/nIBOP/asym-lightgcn-reproducibility.git
git push -u origin main
```

Alternatively, after creating the empty GitHub repository, run:

```powershell
.\PUBLISH_TO_GITHUB.ps1
```

After creating a Zenodo DOI, update:

- `README.md`;
- `CITATION.cff`;
- the article reproducibility paragraph;
- `paper/REPRODUCIBILITY_MANIFEST_RU.md`.
