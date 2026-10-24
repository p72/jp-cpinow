"""Skill module loaders (estat-api / boj-api / fred-data) and polite HTTP.

The skill modules are imported from vendor/<skill>/scripts (bundled copies, used on
GitHub Actions), else from <SKILL>_SKILL_DIR or an installed skill directory.
e-Stat needs the env var ESTAT_APP_ID.
"""
from __future__ import annotations

import glob
import os
import sys
import time
import urllib.request


def _find_skill(name: str, module_file: str) -> str:
    env = os.environ.get(name.upper().replace("-", "_") + "_SKILL_DIR")
    cands = [env] if env else []
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands += [os.path.join(here, "vendor", name)]
    cands += sorted(glob.glob(f"/mnt/skills/*/{name}"))
    cands += sorted(glob.glob(os.path.expanduser(f"~/.claude/skills/{name}")))
    cands += sorted(glob.glob(os.path.expanduser(f"~/.claude/skills/synced/*/{name}")))
    for c in cands:
        if c and os.path.exists(os.path.join(c, "scripts", module_file)):
            return c
    raise ImportError(f"{name} skill not found. Set {name.upper().replace('-', '_')}_SKILL_DIR.")


def _load_skill_module(name: str, module_file: str):
    """Import <skill>/scripts/<module_file> as a package member (relative imports work)."""
    import importlib.util
    import types

    pkg = "_skill_" + name.replace("-", "_")
    modname = f"{pkg}.{module_file[:-3]}"
    if modname in sys.modules:
        return sys.modules[modname]
    sdir = os.path.join(_find_skill(name, module_file), "scripts")
    if pkg not in sys.modules:
        p = types.ModuleType(pkg)
        p.__path__ = [sdir]
        sys.modules[pkg] = p
    spec = importlib.util.spec_from_file_location(modname, os.path.join(sdir, module_file))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def estat():
    return _load_skill_module("estat-api", "estat_api.py")


def boj():
    return _load_skill_module("boj-api", "boj_api.py")


UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def http_get(url: str, tries: int = 3, sleep: float = 1.5, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            time.sleep(sleep)
            return body
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))
