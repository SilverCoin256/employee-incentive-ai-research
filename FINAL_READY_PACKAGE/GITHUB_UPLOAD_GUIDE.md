# GitHub Upload Guide

This guide explains exactly what to upload to GitHub and what not to upload.

## Recommended repository name

Use this repository name:

`constrained-hr-analytics-auditing-pipeline`

## Recommended visibility

Use a **public repository** if you want reviewers, admissions readers, journal editors, or collaborators to see the project easily.

Use a **private repository** only if you are not ready for public release or if you still need to remove personal information.

## Files and folders to upload

Upload only the clean research package and core documentation:

- **`README.md`**: Project overview and how to read the repository.
- **`final_submission_manuscript.tex`**: Main LaTeX manuscript source.
- **`references_verified.bib`**: Verified bibliography file.
- **`submission_ready_manuscript.md`**: Markdown manuscript version.
- **`cover_letter.md`**: Journal cover letter.
- **`supplementary_appendix.md`**: Supplementary appendix.
- **`figures/`**: Final manuscript figures.
- **`tables/`**: Final manuscript tables.
- **`GITHUB_UPLOAD_GUIDE.md`**: This guide.
- **`JOURNAL_SUBMISSION_GUIDE.md`**: Journal submission instructions.

If the final PDF and DOCX have been exported, also upload:

- **`FINAL_POLISHED_MANUSCRIPT.pdf`**
- **`FINAL_POLISHED_MANUSCRIPT.docx`**

## Files and folders not to upload

Do not upload working, cache, or development files:

- **`.venv/`** or any virtual environment folder.
- **`__pycache__/`** folders.
- **`.DS_Store`** files.
- **Temporary exports** or placeholder files.
- **Raw private data** if any private data is ever added later.
- **Unreviewed drafts** that conflict with the final manuscript.
- **Large intermediate model outputs** unless they are necessary for reproducibility.

## Beginner-friendly upload steps

1. Go to <https://github.com/> and sign in.
2. Click the **+** button in the top-right corner.
3. Select **New repository**.
4. Enter the repository name: `constrained-hr-analytics-auditing-pipeline`.
5. Choose **Public** unless you want the project hidden.
6. Do not add a separate GitHub README if you already have the final `README.md` file ready.
7. Click **Create repository**.
8. On the new repository page, choose **uploading an existing file**.
9. Drag in the final package files listed above.
10. Wait until all files finish uploading.
11. In the commit message box, write: `Initial publication-ready submission package`.
12. Click **Commit changes**.

## Suggested README structure

The repository README should include:

- **Project title**
- **One-paragraph summary**
- **Manuscript files**
- **Figures and tables**
- **Reproducibility note**
- **Data availability note**
- **Citation note**
- **Ethical-use warning**

## Final GitHub check

Before sharing the GitHub link, confirm:

- **No placeholder PDF/DOCX files are uploaded.**
- **The final manuscript PDF opens correctly.**
- **The final DOCX opens correctly.**
- **Figures display in the repository.**
- **The bibliography file is present.**
- **No personal/private files are included.**
