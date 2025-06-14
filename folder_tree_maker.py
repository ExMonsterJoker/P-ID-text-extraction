import os
import json

def scan_folder_tree(path, exclude_folders=None):
    if exclude_folders is None:
        exclude_folders = set()

    tree = {}
    for entry in os.scandir(path):
        if entry.is_dir():
            if entry.name in exclude_folders:
                continue
            tree[entry.name] = scan_folder_tree(entry.path, exclude_folders)
        else:
            tree[entry.name] = None
    return tree

def print_tree(tree, indent=0):
    for name, sub in tree.items():
        print('  ' * indent + f"- {name}")
        if isinstance(sub, dict):
            print_tree(sub, indent + 1)

def tree_to_text(tree, indent=0):
    lines = []
    for name, sub in tree.items():
        lines.append('  ' * indent + f"- {name}")
        if isinstance(sub, dict):
            lines.extend(tree_to_text(sub, indent + 1))
    return lines

# Set source path and folders to exclude
source_path = os.getcwd()
excluded = {'.venv',
            '.git',
            '.idea',
            '.pytest_cache',
            '.qodo',
            'tests',
            '__pycache__',
            'tiles',
            'cropping',
            'test',
            'pdf',
            'debug_tile_visualizations',
            'detection_metadata'
            }

folder_tree = scan_folder_tree(source_path, exclude_folders=excluded)


# print(f"Folder tree for: {source_path} (excluding {excluded})")
# print_tree(folder_tree)

# Write tree structure as plain text
# text_lines = [f"Folder tree for: {source_path} (excluding {excluded})"] + tree_to_text(folder_tree)
# with open("folder_tree.txt", "w", encoding="utf-8") as txt_file:
#    txt_file.write('\n'.join(text_lines))

# Write raw structure as JSON
with open("folder_tree.json", "w", encoding="utf-8") as json_file:
    json.dump(folder_tree, json_file, indent=2)

print("Folder tree saved to 'folder_tree.txt' and 'folder_tree.json'")