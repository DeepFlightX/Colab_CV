import os

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