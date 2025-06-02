import os

def scan_folder_tree(path, exclude_folders=None):
    if exclude_folders is None:
        exclude_folders = set()

    tree = {}
    for entry in os.scandir(path):
        if entry.is_dir():
            if entry.name in exclude_folders:
                continue  # Skip excluded folder
            tree[entry.name] = scan_folder_tree(entry.path, exclude_folders)
        else:
            tree[entry.name] = None
    return tree

def print_tree(tree, indent=0):
    for name, sub in tree.items():
        print('  ' * indent + f"- {name}")
        if isinstance(sub, dict):
            print_tree(sub, indent + 1)

# Set source path and folders to exclude
source_path = os.getcwd()
excluded = {'.venv', '.git', '.idea', '.pytest_cache','.qodo','tests', '__pycache__'}  # set of folder names to exclude

folder_tree = scan_folder_tree(source_path, exclude_folders=excluded)
print(f"Folder tree for: {source_path} (excluding {excluded})")
print_tree(folder_tree)
