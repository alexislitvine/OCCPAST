# Summary: What Needs to Change to Use Upstream histocc

This document provides a concise summary of the changes required to use the upstream histocc from the OccCANINE repository.

## Quick Answer

To use the upstream histocc from https://github.com/christianvedels/OccCANINE instead of maintaining a local copy, you need to:

### 1. Convert histocc to a Git Submodule

Replace the local `histocc/` directory with a git submodule pointing to the upstream OccCANINE repository:

```bash
# Remove the local histocc directory
rm -rf histocc

# Add OccCANINE as a submodule
git submodule add https://github.com/christianvedels/OccCANINE.git OccCANINE_upstream

# Checkout the specific commit
cd OccCANINE_upstream
git checkout 844e6be6aa08c00235094ac3cd42698c9cf0c09b
cd ..

# Create a symbolic link to histocc
ln -s OccCANINE_upstream/histocc histocc

# Stage and commit the changes
git add .gitmodules OccCANINE_upstream histocc
git commit -m "Convert histocc to upstream submodule"
```

### 2. Move OCCPAST-Specific Files

Two files existed only in your local copy and need to be preserved:
- `histocc/utils/data_conversion.py` (CSV to Parquet conversion utilities)
- `histocc/utils/descriptions.py` (Occupational code description loading)

Move these to a new module:

```bash
mkdir -p occpast_extensions
cp histocc/utils/data_conversion.py occpast_extensions/
cp histocc/utils/descriptions.py occpast_extensions/
```

Create `occpast_extensions/__init__.py` to export the functions:

```python
from .data_conversion import (
    convert_csv_to_parquet,
    convert_directory_to_parquet,
    get_hisco_dtype_overrides,
    get_hisco_converters,
    get_occ1950_dtype_overrides,
    get_occ1950_converters,
)

from .descriptions import (
    load_hisco_descriptions,
    load_descriptions_from_csv,
    load_descriptions_from_dataframe,
    load_descriptions_from_dict,
    get_hisco_description,
    get_description,
    create_code_to_description_mapping,
    add_descriptions_to_dataframe,
    format_input_with_description,
)
```

### 3. Update Imports

Change all imports from `histocc.utils.*` to `occpast_extensions`:

**Before:**
```python
from histocc.utils.data_conversion import convert_csv_to_parquet
from histocc.utils.descriptions import load_hisco_descriptions
```

**After:**
```python
from occpast_extensions import convert_csv_to_parquet
from occpast_extensions import load_hisco_descriptions
```

Files to update:
- `tests/test_parquet_support.py`
- `tests/test_descriptions.py`
- Any other files that import these modules

### 4. Update Documentation

Update `README.md` to document:
- That histocc is now a submodule
- How to initialize the submodule when cloning
- The existence of `occpast_extensions` for OCCPAST-specific features

## Why These Changes?

1. **Avoid Divergence**: Using a submodule ensures your code stays in sync with upstream OccCANINE
2. **Easy Updates**: Update histocc by simply checking out a new commit in the submodule
3. **Clear Separation**: OCCPAST-specific code is clearly separated from upstream code
4. **Transparency**: The exact version of histocc is tracked via git submodule

## Repository Setup After These Changes

Users cloning the repository will need to initialize the submodule:

```bash
git clone https://github.com/alexislitvine/OCCPAST.git
cd OCCPAST
git submodule update --init --recursive
cd OccCANINE_upstream
git checkout 844e6be6aa08c00235094ac3cd42698c9cf0c09b
cd ..
```

## Files Changed

### Added:
- `.gitmodules` - Submodule configuration
- `OccCANINE_upstream/` - Submodule directory
- `occpast_extensions/` - New module for OCCPAST-specific extensions
- `histocc` - Symbolic link to `OccCANINE_upstream/histocc`

### Modified:
- `README.md` - Updated structure documentation
- `tests/test_parquet_support.py` - Updated imports
- `tests/test_descriptions.py` - Updated imports

### Removed:
- Old `histocc/` directory contents (now provided by submodule)

## Complete Implementation

All these changes have been implemented in this pull request. The repository is now configured to use the upstream histocc from OccCANINE while preserving all OCCPAST-specific functionality in the `occpast_extensions` module.
