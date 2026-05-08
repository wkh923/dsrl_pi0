"""SpaceMouse driver + test scripts for AIRBOT Play single/dual-arm.

Public API is re-exported here so existing callers keep working:

    sys.path.insert(0, ".../Airbot")
    from spacemouse import AirbotSpaceMouse, create_dual_spacemouse, ...

The driver itself lives in ``spacemouse.py``; the sibling ``test_*.py``
scripts are runnable test entry points (run them directly).
"""

from .spacemouse import (  # noqa: F401
    AirbotSpaceMouse,
    DualAirbotSpaceMouse,
    MockDualSpaceMouse,
    MockSpaceMouse,
    SpaceMouseState,
    create_dual_spacemouse,
    create_spacemouse,
)
