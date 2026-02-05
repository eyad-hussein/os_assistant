import os
import argparse
from datetime import datetime


def display_directory_structure(
    path, max_depth=None, include_files=True, output_file=None
):
    """
    Display the directory structure in a tree-like format.

    Args:
        path (str): The directory path to visualize
        max_depth (int, optional): Maximum depth to traverse
        include_files (bool): Whether to include files in the output
        output_file (str, optional): Path to save the tree to a file
    """
    path = os.path.abspath(path)

    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist.")
        return

    result = []
    result.append(f"Directory structure of: {path}")
    result.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    result.append("=" * 50)

    base_name = os.path.basename(path)
    result.append(base_name + "/")

    _scan_directory(path, result, "", max_depth, include_files, 0)

    tree_output = "\n".join(result)

    # Print to console
    print(tree_output)

    # Save to file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(tree_output)
        print(f"\nDirectory structure saved to: {output_file}")


def _scan_directory(path, result, prefix, max_depth, include_files, current_depth):
    """
    Recursively scan directory and build the tree structure.

    Args:
        path (str): Current path to scan
        result (list): List to accumulate the result lines
        prefix (str): Prefix for the current line (for indentation)
        max_depth (int): Maximum depth to traverse
        include_files (bool): Whether to include files
        current_depth (int): Current recursion depth
    """
    if max_depth is not None and current_depth >= max_depth:
        return

    # Get all entries in the directory
    try:
        entries = os.listdir(path)
        entries.sort()

        # Process directories first, then files
        dirs = []
        files = []

        for entry in entries:
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                dirs.append(entry)
            elif include_files:
                files.append(entry)

        # Total count for determining if an item is the last in its list
        total = len(dirs)
        if include_files:
            total += len(files)

        count = 0

        # Process directories
        for dir_name in dirs:
            count += 1
            full_path = os.path.join(path, dir_name)

            # Determine if this is the last item at this level
            is_last = count == total

            # Choose the appropriate prefix characters
            if is_last:
                result.append(f"{prefix}└── {dir_name}/")
                new_prefix = prefix + "    "
            else:
                result.append(f"{prefix}├── {dir_name}/")
                new_prefix = prefix + "│   "

            # Recursively process the subdirectory
            _scan_directory(
                full_path,
                result,
                new_prefix,
                max_depth,
                include_files,
                current_depth + 1,
            )

        # Process files
        if include_files:
            for file_name in files:
                count += 1
                is_last = count == total

                if is_last:
                    result.append(f"{prefix}└── {file_name}")
                else:
                    result.append(f"{prefix}├── {file_name}")

    except PermissionError:
        result.append(f"{prefix}└── [Permission Denied]")
    except Exception as e:
        result.append(f"{prefix}└── [Error: {str(e)}]")


def get_directory_statistics(path):
    """
    Get statistics about the directory structure.

    Args:
        path (str): The directory path to analyze

    Returns:
        dict: Dictionary containing statistics
    """
    path = os.path.abspath(path)

    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist.")
        return None

    stats = {
        "total_dirs": 0,
        "total_files": 0,
        "total_size": 0,
        "max_depth": 0,
        "file_extensions": {},
    }

    _calculate_statistics(path, stats, 0)

    # Format file size to be more readable
    total_size = stats["total_size"]
    if total_size < 1024:
        stats["size_formatted"] = f"{total_size} bytes"
    elif total_size < 1024 * 1024:
        stats["size_formatted"] = f"{total_size / 1024:.2f} KB"
    elif total_size < 1024 * 1024 * 1024:
        stats["size_formatted"] = f"{total_size / (1024 * 1024):.2f} MB"
    else:
        stats["size_formatted"] = f"{total_size / (1024 * 1024 * 1024):.2f} GB"

    return stats


def _calculate_statistics(path, stats, current_depth):
    """
    Recursively calculate statistics for the directory.

    Args:
        path (str): Current path to scan
        stats (dict): Dictionary to accumulate statistics
        current_depth (int): Current recursion depth
    """
    stats["max_depth"] = max(stats["max_depth"], current_depth)

    try:
        entries = os.listdir(path)

        for entry in entries:
            full_path = os.path.join(path, entry)

            if os.path.isdir(full_path):
                stats["total_dirs"] += 1
                _calculate_statistics(full_path, stats, current_depth + 1)
            else:
                stats["total_files"] += 1
                try:
                    stats["total_size"] += os.path.getsize(full_path)

                    # Track file extensions
                    _, ext = os.path.splitext(entry)
                    if ext:
                        ext = ext.lower()
                        stats["file_extensions"][ext] = (
                            stats["file_extensions"].get(ext, 0) + 1
                        )
                    else:
                        stats["file_extensions"]["(no extension)"] = (
                            stats["file_extensions"].get("(no extension)", 0) + 1
                        )
                except:
                    pass
    except:
        pass


def display_statistics(stats):
    """
    Display directory statistics in a readable format.

    Args:
        stats (dict): Dictionary containing statistics
    """
    if not stats:
        return

    print("\nDirectory Statistics:")
    print("=" * 50)
    print(f"Total directories: {stats['total_dirs']}")
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {stats['size_formatted']}")
    print(f"Maximum directory depth: {stats['max_depth']}")

    # Display file extension statistics if there are files
    if stats["total_files"] > 0:
        print("\nFile extensions:")
        sorted_extensions = sorted(
            stats["file_extensions"].items(), key=lambda x: x[1], reverse=True
        )
        for ext, count in sorted_extensions:
            print(f"  {ext}: {count} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display directory structure")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Directory path to visualize (default: current directory)",
    )
    parser.add_argument("--max-depth", type=int, help="Maximum depth to display")
    parser.add_argument(
        "--no-files", action="store_true", help="Don't show files, only directories"
    )
    parser.add_argument("--output", help="Save directory structure to this file")
    parser.add_argument(
        "--stats", action="store_true", help="Show directory statistics"
    )

    args = parser.parse_args()

    # Display directory structure
    display_directory_structure(
        args.path,
        max_depth=args.max_depth,
        include_files=not args.no_files,
        output_file=args.output,
    )

    # Display statistics if requested
    if args.stats:
        stats = get_directory_statistics(args.path)
        display_statistics(stats)
