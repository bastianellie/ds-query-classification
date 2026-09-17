#!/usr/bin/env python
"""Extract a categories.json Category from a free-text instruction as a plain script:

    python extract_categories.py --prompt-file instructions.txt --category-name sentiment

Works whether or not the package is installed: the ``src`` directory is added
to ``sys.path`` so the import below resolves either way.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from query_classification.category_extraction import main  # noqa: E402

if __name__ == "__main__":
    main()
