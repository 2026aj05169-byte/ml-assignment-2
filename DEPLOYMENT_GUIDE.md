# Start here - run, deploy, and submit

## 1. What is ready and what you must supply
The ZIP includes the runnable app, five trained models, all source code, CSVs, metrics, and README. No GitHub repository or live cloud app has been created by this package. You supply your GitHub account/repository, Streamlit login, student details, and a genuine BITS Virtual Lab screenshot. No cloud subscription or API keys are needed for this implementation.

The assignment requires Streamlit Community Cloud specifically. It also requires execution in BITS Virtual Lab. Running only on your Mac or Colab does not satisfy the BITS Lab requirement. The brief says AI is for learning support only: understand and adapt this starter before using it for your coursework.

## 2. Run locally on Mac or Linux
Install Python 3.12 if necessary. Extract ML_Assignment_2_Runnable.zip, open a terminal, and enter the extracted ml_assignment_2 directory.

```bash
cd ml_assignment_2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python verify.py
python -m streamlit run app.py
```

Open http://localhost:8501. If port 8501 is busy, append --server.port 8502. Stop with Ctrl+C. For later runs, activate .venv and run the final command. Internet is required to install packages; dataset loading and trained-model inference work without network downloads.

## 3. Run on Windows / BITS Virtual Lab
Transfer and extract the project in your assigned BITS Virtual Lab desktop. Open a terminal in that directory. Use the lab's available Python 3.12 launcher; on a Windows lab this is commonly:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe train.py
.venv\Scripts\python.exe verify.py
.venv\Scripts\python.exe -m streamlit run app.py
```

If the lab is Linux, use the Mac/Linux commands above. If Python 3.12 is unavailable or internet/install access is blocked, ask BITS Lab support to resolve it; do not silently change Python/package versions. If versions must change, retrain and verify under the new environment and use exactly that same environment on the cloud.

Open the app in the lab browser. Upload test_data.csv, select a model, and confirm all six metrics and the comparison table. Take one genuine screenshot showing the lab environment and assignment execution (e.g., lab desktop/browser with the app, plus terminal training output if it fits legibly). Keep personal credentials out of the screenshot. Save as bits_lab.png. No lab screenshot is included in this package because it must come from your real execution.

If the lab fails, the assignment gives these support contacts: csislabsupport@wilp.bits-pilani.ac.in and raj.rohan@wilp.bits-pilani.ac.in; subject: NSP1 ML Assignment 2: BITS Lab issue. This package does not send email.

## 4. Review and personalize
Read LEARNING_NOTES.md. Explain why scaling is required, why the split is stratified, why malignant is positive, and why AUC needs probabilities. Customize the app wording and your observations meaningfully. If you change the model setup or split, rerun training, update every table/observation in README, and verify again. Do not change seeds repeatedly to seek a better test result. Keep genuine development commits.

## 5. Create your GitHub repository
1. Sign in to https://github.com and create a new public repository, for example ml-assignment-2-cellscope.
2. Upload the CONTENTS of ml_assignment_2, not the enclosing folder. app.py, common.py, requirements.txt, README.md, test_data.csv, and model/ must be at the repository root.
3. Include every .joblib file, model/metadata.json, train.py, train_data.csv, reports/metrics.csv, and the remaining source/docs. Do not upload .venv, caches, passwords, or the ZIP itself. If browser upload omits the hidden .streamlit folder, create .streamlit/config.toml manually or use git; the theme is optional.
4. Commit with an accurate message. Copy the repository URL.
5. Edit README.md: replace REPLACE_WITH_GITHUB_REPOSITORY_URL with the real URL.

Optional command-line upload after creating an empty repository:

```bash
git init
git add .
git commit -m "Add classifier experiment and Streamlit evaluation app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace YOUR_USERNAME and YOUR_REPOSITORY. Authentication is handled by GitHub; never paste tokens into project files.

## 6. Deploy on Streamlit Community Cloud
1. Open https://share.streamlit.io and sign in with GitHub. Allow access to the repository.
2. Choose Create app / New app, then deploy from your GitHub repository.
3. Repository: your repository. Branch: main. Main file path: app.py.
4. Open Advanced settings and explicitly select Python 3.12 to match the saved models. Keep requirements.txt at the root.
5. Choose an available app subdomain if offered, then click Deploy.
6. Wait for the build logs to finish. Open the resulting https://YOUR_APP.streamlit.app URL.
7. Download the sample test CSV from the sidebar, upload it back, select each of the five models, and check metrics, confusion matrix, and comparison table. Test the public link in an incognito window so the evaluator can open it.
8. Replace REPLACE_WITH_STREAMLIT_APP_URL in README.md with the actual live URL and commit the updated README.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
Dependencies: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

The cloud only runs app.py and loads saved artifacts. There is no training step on deployment. An app may sleep when inactive; open it again before final checking.

## 7. Generate the final submission PDF
Do this after BOTH links work and your real lab screenshot is saved. The generator refuses missing links, missing screenshot, or a README with unresolved markers. It does not verify remote link availability, so check the links manually first.

Install the optional report package:

```bash
python -m pip install -r requirements-report.txt
```

Update README.md with your real URLs first, then run the following as one command (replace every YOUR value):

```bash
python build_submission.py --name "YOUR NAME" --student-id "YOUR BITS ID" --github "https://github.com/YOUR_USERNAME/YOUR_REPOSITORY" --app "https://YOUR_APP.streamlit.app" --screenshot "bits_lab.png" --output "submission.pdf"
```

The PDF contains the required order: (1) clickable GitHub link, (2) clickable live app link, (3) your BITS Lab screenshot, (4) full README content. Open it and check all pages, links, screenshot readability, and tables. Ensure README in GitHub and PDF match. The report generator can run on your Mac if preferred; the assignment execution evidence still must come from BITS Lab.

## 8. Final checks and submit
- GitHub opens publicly and includes all five .joblib models, source, requirements, README, and test CSV.
- Live app opens interactively and evaluates uploads without training.
- App metrics match reports/metrics.csv for bundled test_data.csv.
- README and PDF show your real URLs and your own understood observations.
- PDF includes a genuine BITS Lab screenshot and the complete README.
- Upload a SINGLE PDF to the course portal and press SUBMIT, not Save Draft. Download it back to confirm the submitted file.

The brief states 8 November 2026, 23:59, but does not explicitly identify a timezone. Plan conservatively using IST: 8 November 23:59 IST is 8 November 22:29 UAE. Confirm the actual portal cutoff and submit well before it. Do not depend on an extension.

## Troubleshooting
| Symptom | Fix |
| --- | --- |
| ModuleNotFoundError | Activate the virtual environment; run python -m pip install -r requirements.txt. |
| Cannot load model / version warning | Match Python 3.12 and pinned packages; ensure all model files are in GitHub. Retrain only in the matching training environment if needed. |
| Cloud cannot find app.py | Move project contents to repository root or correct the main-file path. |
| CSV rejected | Use supplied headers and numeric raw values, plus target with 0/1 labels. No ID or extra index column. |
| AUC N/A | The uploaded target contains only one class. Upload a labeled test set containing both classes. |
| Missing model file after upload | Upload the whole model directory with all five .joblib files and metadata.json. |
| Report generation fails | Install requirements-report.txt; replace README markers; supply real valid URLs and a PNG/JPG screenshot. |
| Permission errors installing in lab | Use a project virtual environment; ask BITS support if lab policy blocks installation. |
