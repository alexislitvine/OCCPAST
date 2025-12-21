# Migration to Upstream histocc

This document explains the changes made to use the upstream histocc from the OccCANINE repository instead of maintaining a local copy.

## What Changed

### 1. histocc Module Source

**Before:**
- `histocc/` was a regular directory containing a copy of the histocc code
- Local modifications were made directly to this copy

**After:**
- `histocc/` is now a symbolic link to `OccCANINE_upstream/histocc`
- The upstream OccCANINE repository is included as a git submodule at `OccCANINE_upstream`
- The submodule is pinned to commit `844e6be6aa08c00235094ac3cd42698c9cf0c09b`

### 2. OCCPAST-Specific Extensions

Two files that were custom to OCCPAST have been moved to a new module:

**Before:**
- `histocc/utils/data_conversion.py`
- `histocc/utils/descriptions.py`

**After:**
- `occpast_extensions/data_conversion.py`
- `occpast_extensions/descriptions.py`
- `occpast_extensions/__init__.py` (new module interface)

### 3. Import Changes

All imports of the OCCPAST-specific functionality have been updated:

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

### 4. Test Updates

The following test files have been updated to use the new imports:
- `tests/test_parquet_support.py`
- `tests/test_descriptions.py`

## Benefits

1. **Avoid Divergence**: By using the upstream histocc as a submodule, we avoid the code diverging from the original OccCANINE repository.

2. **Easy Updates**: To update histocc to a newer version from upstream:
   ```bash
   cd OccCANINE_upstream
   git fetch origin
   git checkout <new-commit-hash>
   cd ..
   git add OccCANINE_upstream
   git commit -m "Update histocc to <new-commit-hash>"
   ```

3. **Clear Separation**: OCCPAST-specific functionality is clearly separated in the `occpast_extensions` module, making it easy to identify what's custom vs. what comes from upstream.

4. **Transparency**: The exact version of histocc being used is tracked via the submodule commit hash.

## Repository Setup

When cloning this repository, you must initialize the submodule:

```bash
git clone https://github.com/alexislitvine/OCCPAST.git
cd OCCPAST
git submodule update --init --recursive
cd OccCANINE_upstream
git checkout 844e6be6aa08c00235094ac3cd42698c9cf0c09b
cd ..
```

## Submodule Management

### Updating to a Newer Upstream Version

To update histocc to use a newer version from upstream OccCANINE:

```bash
cd OccCANINE_upstream
git fetch origin
git checkout <commit-hash-or-branch>
cd ..
git add OccCANINE_upstream
git commit -m "Update histocc to upstream commit <commit-hash>"
```

### Checking Current Version

To see which commit of histocc is currently being used:

```bash
cd OccCANINE_upstream
git log -1
```

Or from the root:

```bash
git submodule status
```

## Files Modified

### New Files
- `.gitmodules` - Git submodule configuration
- `OccCANINE_upstream/` - Git submodule directory
- `occpast_extensions/__init__.py` - New module for OCCPAST extensions
- `occpast_extensions/data_conversion.py` - Moved from histocc/utils/
- `occpast_extensions/descriptions.py` - Moved from histocc/utils/
- `MIGRATION_TO_UPSTREAM.md` - This documentation

### Modified Files
- `README.md` - Updated to reflect new structure
- `tests/test_parquet_support.py` - Updated imports
- `tests/test_descriptions.py` - Updated imports
- `histocc` - Now a symbolic link instead of a directory

### Removed Files
- Previous `histocc/` directory contents (now provided by submodule)
- `histocc/utils/data_conversion.py` (moved to occpast_extensions)
- `histocc/utils/descriptions.py` (moved to occpast_extensions)

## Compatibility Notes

All existing functionality is preserved. The changes are purely organizational:
- All histocc imports continue to work as before
- OCCPAST-specific utilities are now imported from `occpast_extensions`
- Tests continue to pass with the new structure
