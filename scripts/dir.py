import os
import shutil

def list_dir(path):
    """Return a list of top-level files and folders in the given directory."""
    try:
        return os.listdir(path)
    except FileNotFoundError:
        print(f"Error: Path '{path}' not found.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for '{path}'.")
        return []

def find_extra_item(list1, list2):
    """
    Given two lists where one has exactly one extra element,
    return that extra element's name.
    """
    set1 = set(list1)
    set2 = set(list2)

    # Find which list is bigger and return the difference
    if len(list1) > len(list2):
        diff = set1 - set2
    else:
        diff = set2 - set1

    return diff.pop() if diff else None

def copy_folder_if_exists(source_dir, folder_name, destination_dir):
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    target_folder_path = source_dir / folder_name

    if target_folder_path.exists() and target_folder_path.is_dir():
        dest_path = destination_dir / folder_name
        try:
            shutil.copytree(target_folder_path, dest_path)
            print(f" Folder '{folder_name}' copied to '{destination_dir}'.")
        except FileExistsError:
            print(f" Destination folder '{dest_path}' already exists.")
        except Exception as e:
            print(f" Error copying folder: {e}")
    else:
        print(f" Folder '{folder_name}' not found in '{source_dir}'.")