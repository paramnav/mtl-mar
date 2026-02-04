#!/usr/bin/env python3
"""
Script to fix hardcoded paths in Jupyter notebooks.
Replaces all instances of /gxfs_work/cau/sunms498/work/nn4sedratetoc/ with relative paths using utils.
"""
import json
import sys
from pathlib import Path

def fix_notebook_paths(notebook_path):
    """Fix hardcoded paths in a Jupyter notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes_made = False
    
    # Path replacements
    replacements = [
        ('/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/raw/', 'DATA_RAW / '),
        ('/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/', 'DATA_INTERIM / '),
        ('/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/output/', 'DATA_OUTPUT / '),
        ('/gxfs_work/cau/sunms498/work/nn4sedratetoc/src/', 'PROJECT_ROOT / "src" / '),
        ('Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/raw/', 'DATA_RAW / '),
        ('Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/', 'DATA_INTERIM / '),
        ('Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/output/', 'DATA_OUTPUT / '),
    ]
    
    # Check if first cell has utils import
    has_utils_import = False
    utils_import_code = '''from pathlib import Path
import sys

# Get project root and add to path
project_root = Path().resolve().parents[0] if Path().resolve().name == 'notebooks' else Path().resolve()
if project_root.name != 'mtl-mar':
    project_root = project_root.parent
sys.path.insert(0, str(project_root / "src"))
from utils import DATA_RAW, DATA_INTERIM, DATA_OUTPUT, PROJECT_ROOT
'''
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Check if utils already imported
            if 'from utils import' in source or 'DATA_RAW' in source:
                has_utils_import = True
            
            # Fix paths in source
            original_source = source
            for old_path, new_path in replacements:
                if old_path in source:
                    # Handle Path() wrapper
                    if source.startswith('Path("') or 'Path("/gxfs_work' in source:
                        source = source.replace(f'Path("{old_path}', f'{new_path}"')
                        source = source.replace(f'Path(\'{old_path}', f'{new_path}\'')
                    else:
                        source = source.replace(old_path, new_path)
                    changes_made = True
            
            # Fix specific patterns
            if 'Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated/")' in source:
                source = source.replace(
                    'Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated/")',
                    'DATA_RAW / "features" / "FeaturesPhrampusLee_TOCSedRate_updated"'
                )
                changes_made = True
            
            if 'Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/selectedFeatureLists/selectedfeatures_v_men.txt")' in source:
                source = source.replace(
                    'Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/selectedFeatureLists/selectedfeatures_v_men.txt")',
                    'DATA_INTERIM / "selectedFeatureLists" / "selectedfeatures_v_men.txt"'
                )
                changes_made = True
            
            if 'Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_DO_men")' in source:
                source = source.replace(
                    'Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_DO_men")',
                    'DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_MAR_DO_men"'
                )
                changes_made = True
            
            if 'Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_predict_men/")' in source:
                source = source.replace(
                    'Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_predict_men/")',
                    'DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"'
                )
                changes_made = True
            
            if 'Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_explain/")' in source:
                source = source.replace(
                    'Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc/data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_explain/")',
                    'DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_explain"'
                )
                changes_made = True
            
            if source != original_source:
                # Split source back into lines
                cell['source'] = source.splitlines(keepends=True)
                if not cell['source'][-1].endswith('\n'):
                    cell['source'][-1] += '\n'
    
    # Add utils import to first code cell if not present
    if not has_utils_import and changes_made:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code' and len(cell['source']) > 0:
                # Check if it's an import cell
                source = ''.join(cell['source'])
                if 'import' in source.lower() and ('numpy' in source or 'pandas' in source or 'pathlib' in source):
                    # Prepend utils import
                    existing_imports = ''.join(cell['source'])
                    cell['source'] = (utils_import_code + '\n' + existing_imports).splitlines(keepends=True)
                    break
    
    if changes_made:
        # Backup original
        backup_path = notebook_path.with_suffix('.ipynb.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        # Write fixed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        print(f"Fixed: {notebook_path.name}")
        return True
    
    return False

if __name__ == '__main__':
    notebooks_dir = Path(__file__).parent / 'notebooks'
    
    if not notebooks_dir.exists():
        print(f"Notebooks directory not found: {notebooks_dir}")
        sys.exit(1)
    
    notebooks = list(notebooks_dir.glob('*.ipynb'))
    fixed_count = 0
    
    for nb_path in notebooks:
        if fix_notebook_paths(nb_path):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} out of {len(notebooks)} notebooks")

