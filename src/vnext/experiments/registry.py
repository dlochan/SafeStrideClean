from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class VNextExperiment:
    """Definition of a single vNext experiment variant."""

    name: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    # Optional free-form description for documentation purposes
    description: Optional[str] = None


@dataclass
class VNextExperimentSuite:
    """Collection of experiments sharing a base configuration."""
    
    base_config_path: str
    experiments: List[VNextExperiment]


def load_experiment_suite(path: str | Path) -> VNextExperimentSuite:
    """Load experiment suite definition from YAML file.
    
    Expected YAML structure:
        base_config: "configs/vnext_example.yaml"
        experiments:
          - name: "experiment_1"
            tags: ["tag1", "tag2"]
            overrides:
              model:
                type: "fz"
              features:
                enable_kinematics: true
          - name: "experiment_2"
            ...
    
    Parameters
    ----------
    path : Path to the experiment suite YAML file
    
    Returns
    -------
    VNextExperimentSuite
        Loaded experiment suite with base config path and experiment list
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment suite file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at root of {path}, got {type(data)}")
    
    base_config = data.get("base_config")
    if not base_config:
        raise ValueError(f"Missing 'base_config' in {path}")
    
    experiments_data = data.get("experiments", [])
    if not isinstance(experiments_data, list):
        raise ValueError(f"'experiments' must be a list in {path}")
    
    experiments: List[VNextExperiment] = []
    for exp_dict in experiments_data:
        if not isinstance(exp_dict, dict):
            raise ValueError(f"Each experiment must be a dict in {path}")
        
        name = exp_dict.get("name")
        if not name:
            raise ValueError(f"Each experiment must have a 'name' in {path}")
        
        overrides = exp_dict.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"'overrides' must be a dict for experiment '{name}' in {path}")
        
        tags = exp_dict.get("tags", [])
        if not isinstance(tags, list):
            raise ValueError(f"'tags' must be a list for experiment '{name}' in {path}")
        
        description = exp_dict.get("description")

        experiments.append(VNextExperiment(
            name=str(name),
            overrides=overrides,
            tags=[str(t) for t in tags],
            description=str(description) if description is not None else None,
        ))
    
    return VNextExperimentSuite(
        base_config_path=str(base_config),
        experiments=experiments,
    )


def deep_merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override dict into base dict, preserving unspecified keys.

    Example
    -------
    >>> base = {"model": {"type": "fz", "hidden": 32}}
    >>> overrides = {"model": {"type": "grf3d"}}
    >>> deep_merge_dicts(base, overrides)
    {'model': {'type': 'grf3d', 'hidden': 32}}

    Notes
    -----
    This function never mutates ``base`` or ``overrides``; it always returns
    a new dictionary.

    Parameters
    ----------
    base : Base configuration dictionary
    overrides : Override dictionary to merge in

    Returns
    -------
    dict
        New dict with overrides applied (base is not modified)
    """
    import copy
    result = copy.deepcopy(base)
    
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = deep_merge_dicts(result[key], value)
        else:
            # Replace value
            result[key] = copy.deepcopy(value)
    
    return result
