#!/usr/bin/env python
"""Run the query classifier as a plain script:

    python classify.py --input data/queries.csv --column text ...

Works whether or not the package is installed: the ``src`` directory is added
to ``sys.path`` so the import below resolves either way.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from query_classification.cli import main  # noqa: E402

if __name__ == "__main__":
    main()