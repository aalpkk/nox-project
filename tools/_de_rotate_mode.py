"""Auto-rotation mode flag for the DE v1 baseline pilots.

When the environment variable ``DE_V1_ROTATE`` is set (non-empty), the base
pilot generators (ema_context_pilot{3,4,5,6,7}.py) skip their locked-baseline
``EXPECTED_*`` row-count / sha256 assertions and build on whatever HB +
ema_context are currently present. This is used ONLY by
``tools/de_v1_rotate_baseline.py`` to regenerate the frozen baseline against a
fresh HB (rotate-on-HALT). In all normal runs the variable is unset, so the
assertions remain fully active — research reproducibility is unchanged.

Version-string checks (scanner_version, breakpoints_version) are NOT bypassed:
those are pipeline-invariant and must hold even during a rotation.
"""

from __future__ import annotations

import os

ROTATE_MODE: bool = bool(os.environ.get("DE_V1_ROTATE"))
