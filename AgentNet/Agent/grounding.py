import os
import fnmatch
from AgentNet.Utils import SSHClient

def load_ignore_patterns(ignore_file=".ignore"):
    """
    Load ignore patterns from the specified .ignore file.
    """
    ignore_patterns = []
    if os.path.exists(ignore_file):
        with open(ignore_file, "r") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):  # Ignore empty lines and comments
                    ignore_patterns.append(line)
    return ignore_patterns

def is_ignored(path, ignore_patterns):
    """
    Check if the given path matches any of the ignore patterns.
    """
    for pattern in ignore_patterns:
        # Match directories with trailing `/`
        if pattern.endswith("/") and os.path.isdir(path) and fnmatch.fnmatch(path, f"*{pattern.rstrip('/')}*"):
            return True
        # Match files or general patterns
        elif fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False


def generate_tree_structure(base_path, ignore_patterns, prefix=""):
    """
    Recursively generate a tree structure as a string, excluding ignored files/folders.
    """
    tree = []  # List to accumulate the tree structure as strings
    items = sorted(os.listdir(base_path))
    
    # Filter out ignored files and folders
    items = [
        item for item in items
        if not is_ignored(os.path.join(base_path, item), ignore_patterns)
    ]
    
    for index, item in enumerate(items):
        full_path = os.path.join(base_path, item)
        connector = "└── " if index == len(items) - 1 else "├── "
        tree.append(f"{prefix}{connector}{item}")
        
        # Recurse into directories, but only if they are not ignored
        if os.path.isdir(full_path):
            new_prefix = prefix + ("    " if index == len(items) - 1 else "│   ")
            tree.append(generate_tree_structure(full_path, ignore_patterns, new_prefix))
    
    # if empty, return Current directory is empty
    if not tree or tree == [""]:
        return "Currently the directory is empty"

    return "\n".join(tree)  # Join all lines into a single string

def generate_ssh_grounding(client: SSHClient):
    """
    Generate SSH grounding for its working directory.
    Assumes the client is already connected and its ready to use
    """
    files = client.execute_command("ls")
    if files == "":
        return "Currently the directory is empty"
    return files
