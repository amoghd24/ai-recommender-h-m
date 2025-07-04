import yaml
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Public API
#   load_feature_config        -> entire YAML as dict
#   get_customer_feature_config -> merged customer features
#   get_article_feature_config -> merged article features
#   get_categorical_sizes      -> map feature_name -> num_categories
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]  # project root
    / "config"
    / "feature_config.yaml"
)


@lru_cache(maxsize=1)
def load_feature_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the YAML feature configuration.

    Results are memoised so the file is read only once per interpreter.
    """
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Feature config not found at {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def _merge_feature_sections(section: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Merge the `categorical` and `numerical` sub-sections for one scope."""
    merged: Dict[str, Dict[str, Any]] = {}
    for sub in ("categorical", "numerical"):
        sub_cfg = section.get(sub, {}) or {}
        merged.update(sub_cfg)
    return merged


def get_customer_feature_config() -> Dict[str, Dict[str, Any]]:
    """Return combined customer feature configuration."""
    cfg = load_feature_config()
    return _merge_feature_sections(cfg["customer"])


def get_article_feature_config() -> Dict[str, Dict[str, Any]]:
    """Return combined article feature configuration."""
    cfg = load_feature_config()
    return _merge_feature_sections(cfg["article"])


def get_categorical_sizes(scope: Optional[str] = None) -> Dict[str, int]:
    """Return mapping of feature_name -> num_categories.

    Args:
        scope: 'customer', 'article', or None for both.
    """
    cfg = load_feature_config()
    scopes = [scope] if scope else ["customer", "article"]
    sizes: Dict[str, int] = {}

    for sc in scopes:
        section = cfg[sc]["categorical"]
        for feat, meta in section.items():
            sizes[f"{sc}_{feat}"] = meta["num_categories"]

    return sizes


# ---------------------------------------------------------------------------
# Legacy variables for backward-compatibility with existing imports
# ---------------------------------------------------------------------------

CUSTOMER_CATEGORICAL_FEATURES = load_feature_config()["customer"]["categorical"]
CUSTOMER_NUMERICAL_FEATURES = load_feature_config()["customer"]["numerical"]
ARTICLE_CATEGORICAL_FEATURES = load_feature_config()["article"]["categorical"]
ARTICLE_NUMERICAL_FEATURES = load_feature_config()["article"]["numerical"]

__all__ = [
    "load_feature_config",
    "get_customer_feature_config",
    "get_article_feature_config",
    "get_categorical_sizes",
    "CUSTOMER_CATEGORICAL_FEATURES",
    "CUSTOMER_NUMERICAL_FEATURES",
    "ARTICLE_CATEGORICAL_FEATURES",
    "ARTICLE_NUMERICAL_FEATURES",
] 