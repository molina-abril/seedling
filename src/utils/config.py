"""Configuration loading and management."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ruamel.yaml import YAML

class ConfigManager:
    """Centralized configuration loader for the entire pipeline."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize config manager.

        Args:
            config_dir: Path to configs directory. If None, defaults to PROJECT_ROOT/configs
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "configs"

        self.config_dir = Path(config_dir)
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self._cache: Dict[str, Any] = {}

    def load_clustering(self, model: str = "bertopic") -> Dict[str, Any]:
        """Load clustering config for a specific model.

        Args:
            model: clustering model name (e.g., 'bertopic')

        Returns:
            Configuration dict for that model.
        """
        return self._load_yaml(f"clustering/{model}.yaml")

    def _load_yaml(self, relative_path: str) -> Dict[str, Any]:
        """Load a YAML file with caching.

        Args:
            relative_path: path relative to config_dir (e.g., 'clustering/bertopic.yaml')

        Returns:
            Parsed YAML content as dict.
        """
        if relative_path in self._cache:
            return self._cache[relative_path]

        filepath = self.config_dir / relative_path
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = self.yaml.load(f)

        if data is None:
            data = {}

        self._cache[relative_path] = data
        return data

