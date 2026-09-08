#!/usr/bin/env python
"""Run the experiment runner as a plain script:

    python experiment.py induce --train-file t.csv --text-column text \
        --label-column label --category-name sentiment --run-dir runs/1

Works whether or not the package is installed: the ``src`` directory is added
to ``sys.path`` so the import below resolves either way.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from query_classification.experiment import main  # noqa: E402

if __name__ == "__main__":
    main()
