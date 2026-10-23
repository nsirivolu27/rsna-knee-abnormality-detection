"""Cells for the Kaggle labeling notebook.

Requirements in the notebook: GPU accelerator, Internet ON (to clone), and a
multilingual instruct model attached through Add Input, Models.

Paste each block below as its own cell.
"""

# --- CELL 1: repository and dependencies -----------------------------------
CELL_1 = r"""
!rm -rf /kaggle/working/repo
!git clone -q https://github.com/nsirivolu27/rsna-knee-abnormality-detection.git /kaggle/working/repo
!pip install -q accelerate bitsandbytes

import sys
sys.path.insert(0, "/kaggle/working/repo")
for module in [m for m in sys.modules if m.startswith(("src", "scripts"))]:
    del sys.modules[module]
"""

# --- CELL 2: load the model -------------------------------------------------
CELL_2 = r"""
from scripts.hf_generator import build_generator, find_model_directory

print("model directory:", find_model_directory())
generate = build_generator(max_new_tokens=256)

# Smoke test one prompt before committing to thousands.
print(generate(["Reply with the single word: ready"])[0][:200])
"""

# --- CELL 3: pilot on the 58 expert examinations ---------------------------
# Always run this before the full corpus. Prompt revisions are cheap at 58
# examinations and expensive at 4,349.
CELL_3 = r"""
import pandas as pd
from src import config
from src.labels import fully_labeled_exams, load_labels
from scripts.run_labeling import main as run_labeling_main

expert = load_labels()
pilot_ids = fully_labeled_exams(expert).index
reports = pd.read_csv(config.TRAIN_CSV, usecols=[config.EXAM_ID_COLUMN, config.REPORT_COLUMN])
pilot = reports[reports[config.EXAM_ID_COLUMN].isin(pilot_ids)]
print("pilot examinations:", len(pilot))

results, failures, agreement = run_labeling_main(
    generate,
    reports=pilot,
    out_path="/kaggle/working/soft_labels_pilot.csv",
    expert_labels=expert,
    batch_size=4,
)
"""

# --- CELL 4: full corpus, only once the pilot looks acceptable -------------
# Resumable. If the session dies, rerun this cell and it continues from the
# checkpoint rather than starting over.
CELL_4 = r"""
import pandas as pd
from src import config
from scripts.run_labeling import main as run_labeling_main

reports = pd.read_csv(config.TRAIN_CSV, usecols=[config.EXAM_ID_COLUMN, config.REPORT_COLUMN])
print("examinations to label:", len(reports))

results, failures, agreement = run_labeling_main(
    generate,
    reports=reports,
    out_path="/kaggle/working/soft_labels_v1.csv",
    batch_size=8,
    checkpoint_every=50,
)
"""
