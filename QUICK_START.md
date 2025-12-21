# Quick Start: Using the New Structure

After the migration to upstream histocc, here's what you need to know to get started.

## For New Users Cloning the Repository

```bash
# Clone the repository
git clone https://github.com/alexislitvine/OCCPAST.git
cd OCCPAST

# Initialize the histocc submodule
git submodule update --init --recursive

# Checkout the specific commit
cd OccCANINE_upstream
git checkout 844e6be6aa08c00235094ac3cd42698c9cf0c09b
cd ..

# Now you're ready to use the repository!
```

## For Existing Users Pulling Updates

```bash
# Pull the latest changes
git pull

# Update submodules
git submodule update --init --recursive

# Ensure correct commit is checked out
cd OccCANINE_upstream
git checkout 844e6be6aa08c00235094ac3cd42698c9cf0c09b
cd ..
```

## Import Changes

If you have any custom scripts that import from histocc, you may need to update them:

### Before:
```python
from histocc.utils.data_conversion import convert_csv_to_parquet
from histocc.utils.descriptions import load_hisco_descriptions
```

### After:
```python
from occpast_extensions import convert_csv_to_parquet
from occpast_extensions import load_hisco_descriptions
```

**Note**: Only imports from `histocc.utils.data_conversion` and `histocc.utils.descriptions` need to change. All other histocc imports remain the same:

```python
# These imports remain unchanged
from histocc import OccCANINE
from histocc.formatter import hisco_blocky5
from histocc.dataloader import OccDatasetMixerInMemMultipleFiles
# etc.
```

## What Changed?

### Directory Structure

**Before:**
```
OCCPAST/
├── histocc/              # Full copy of histocc code
│   ├── utils/
│   │   ├── data_conversion.py
│   │   └── descriptions.py
│   └── ...
└── tests/
```

**After:**
```
OCCPAST/
├── OccCANINE_upstream/   # Git submodule (upstream repo)
│   └── histocc/          # The actual histocc code
├── histocc -> OccCANINE_upstream/histocc  # Symbolic link
├── occpast_extensions/   # OCCPAST-specific code
│   ├── __init__.py
│   ├── data_conversion.py
│   └── descriptions.py
└── tests/
```

## Benefits

1. **Always in sync with upstream**: histocc code comes directly from OccCANINE repository
2. **Easy updates**: Just checkout a newer commit in the submodule
3. **No code duplication**: Single source of truth for histocc
4. **Clear ownership**: OCCPAST-specific code is clearly separated

## Updating histocc to a Newer Version

When a new version of OccCANINE is released:

```bash
cd OccCANINE_upstream
git fetch origin
git log --oneline  # Find the commit you want
git checkout <commit-hash>
cd ..
git add OccCANINE_upstream
git commit -m "Update histocc to upstream commit <commit-hash>"
```

## Common Questions

**Q: Why is histocc a symbolic link?**  
A: To allow Python imports to work seamlessly while histocc code lives in the submodule.

**Q: What if I need to modify histocc code?**  
A: Don't modify code in the submodule. Instead:
- If it's OCCPAST-specific, add it to `occpast_extensions/`
- If it should be in upstream, contribute to the OccCANINE repository

**Q: What happened to the data_conversion and descriptions modules?**  
A: They were OCCPAST-specific additions, so they've been moved to `occpast_extensions/` where they belong.

## Need Help?

- See `MIGRATION_TO_UPSTREAM.md` for detailed information
- See `CHANGES_SUMMARY.md` for a complete list of changes
- Check the updated `README.md` for general repository documentation
