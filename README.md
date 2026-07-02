# twogroup-linalg

Linear algebra utilities for finite two-groups.

## Installation

Install the package in editable mode from this directory:

```bash
python -m pip install -e .
```

The current package layout is preserved as-is, with a top-level package wrapper
for regular imports:

```python
import twogroup_linalg
```

Optional wrappers can be installed with extras:

```bash
python -m pip install -e ".[flint,galois]"
```

Development tools can be installed with:

```bash
python -m pip install -e ".[dev]"
```
