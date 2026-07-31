"""
Pillar 4 — Cython obfuscation: compile proprietary .py into native .pyd modules.

Turns the readable Python for your core IP into C-extensions, so the shipped app
contains compiled machine code instead of source. This is the standard way to
raise the reverse-engineering bar for Python apps.

PREREQS
    pip install cython
    Windows: "Microsoft C++ Build Tools" (MSVC) must be installed.

BUILD
    python setup_obfuscate.py build_ext --inplace

AFTER BUILDING (this is the part that actually protects you)
    1. Embed secrets into the COMPILED modules first:
         - licensing/verify.py        -> set _EMBEDDED_PUBLIC_KEY
         - licensing/secure_loader.py -> set _APP_SECRET
         - src/pipeline.py            -> make the license gate UNCONDITIONAL
    2. Rebuild, then DELETE the matching .py and generated .c files from the
       shipped folder — otherwise the source is still right there next to the .pyd.
    3. Ship a thin launcher that imports the compiled module, e.g. run_app.py:
           import sys; sys.path.insert(0, "src"); import pipeline; pipeline.run(...)
       (you can't `python pipeline.pyd` directly — a .pyd is imported, not run.)

HONEST LIMIT: Cython output is C-compiled but still uses the CPython API; tools
exist to partially recover logic, and the bytecode-level secrets can be dumped
from memory. It deters casual theft and source lifting — it is not encryption.
"""
import sys
from pathlib import Path

try:
    from setuptools import setup
    from Cython.Build import cythonize
except ImportError:
    print("Install build deps first:  pip install cython setuptools")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent

# The proprietary logic worth compiling. Add/remove as you like.
MODULES = [
    "src/pipeline.py",
    "src/tracker.py",
    "src/handball_rules.py",
    "src/goal_post_detector.py",
    "src/ball_kalman.py",
    "src/detector.py",
    "licensing/verify.py",
    "licensing/hardware_id.py",
    "licensing/clock_guard.py",
    "licensing/secure_loader.py",
]

DIRECTIVES = {
    "language_level": 3,
    "annotation_typing": False,   # tolerate runtime annotations / dataclasses
    "always_allow_keywords": True,
}


def main():
    existing = [m for m in MODULES if (ROOT / m).exists()]
    missing = [m for m in MODULES if not (ROOT / m).exists()]
    if missing:
        print("[!] skipping (not found):", ", ".join(missing))
    print("[Cython] compiling:", ", ".join(existing))
    setup(
        name="handball_protected",
        ext_modules=cythonize(existing, compiler_directives=DIRECTIVES,
                              quiet=False),
        script_args=["build_ext", "--inplace"],
    )
    print("\n[Cython] done. Remember: embed secrets, DELETE the .py/.c sources, "
          "ship a thin launcher that imports the .pyd.")


if __name__ == "__main__":
    main()
