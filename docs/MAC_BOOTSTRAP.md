# SafeStrideClean Mac bootstrap

- Repo: `/Volumes/Extreme SSD/safestride_clean`
- Venv: `~/venvs/safestrideclean`

## Python requirement

You need Python **>= 3.10** available on your Mac. A simple way is Homebrew:

```bash
brew install python@3.12
python3.12 --version
```

## Dataset root

`SAFESTRIDE_DATA_ROOT` must point to a directory that contains `datasets/ProcessedData`.
Example on this Mac:

```bash
export SAFESTRIDE_DATA_ROOT="/Volumes/Extreme SSD/safestride"
```

## Bootstrap

From the repo root:

```bash
source .env.local.example
bash scripts/bootstrap_mac.sh
```
