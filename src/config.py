# src/config.py
from pathlib import Path

# Raw dataset root (external SSD)
DATA_ROOT = Path(r"E:\safestride\datasets\ProcessedData")

# Your local working/output roots (on C:)
WORK_ROOT = Path(r"C:\Users\locha\Documents\safestride\data\working")
OUT_ROOT  = Path(r"C:\Users\locha\Documents\safestride\out_grid")

# Defaults
FS_HZ     = 200
BW_BY_SUBJECT = {
    # fill as you go; you already used 78.9 for AB01
    "AB01": 78.9,
    "AB02": 82.2,
    "AB03": 113.5,
    "AB05": 71.5,
    "AB06": 79.1,
    "AB07": 62.3,
    "AB08": 87.6,
    "AB09": 84.1,
    "AB10": 67.5,
    "AB11": 65.1,
    "AB12": 64.0,
    "AB13": 67.6,
}
