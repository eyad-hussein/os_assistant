import os
import shutil
import random
import datetime
import time


class FileSystemOperations:
    def __init__(self, base_path):
        self.base_path = base_path

        # Create base directory if it doesn't exist
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
            print(f"Created base directory: {self.base_path}")

    def create_folder(self, folder_name):
        """Create a new folder"""
        folder_path = os.path.join(self.base_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
            return True
        return False

    def delete_folder(self, folder_name):
        """Delete a folder and its contents"""
        folder_path = os.path.join(self.base_path, folder_name)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
            return True
        return False

    def create_file(self, file_path, content=""):
        """Create a new file with optional content"""
        full_path = os.path.join(self.base_path, file_path)
        dir_name = os.path.dirname(full_path)

        # Create parent directories if they don't exist
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Created file: {full_path}")
        return True

    def delete_file(self, file_path):
        """Delete a file"""
        full_path = os.path.join(self.base_path, file_path)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            os.remove(full_path)
            print(f"Deleted file: {full_path}")
            return True
        return False

    def rename_item(self, old_path, new_path):
        """Rename a file or folder"""
        full_old_path = os.path.join(self.base_path, old_path)
        full_new_path = os.path.join(self.base_path, new_path)

        if os.path.exists(full_old_path):
            # Create parent directories for new path if they don't exist
            new_dir = os.path.dirname(full_new_path)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            shutil.move(full_old_path, full_new_path)
            print(f"Renamed {old_path} to {new_path}")
            return True
        return False

    def edit_file(self, file_path, content=None, append=False):
        """Edit a file's content"""
        full_path = os.path.join(self.base_path, file_path)

        if not os.path.exists(full_path):
            return False

        mode = "a" if append else "w"
        with open(full_path, mode, encoding="utf-8") as f:
            if content:
                f.write(content)

        operation = "Appended to" if append else "Edited"
        print(f"{operation} file: {full_path}")
        return True

    def compress_file(self, file_path, archive_name=None):
        """Compress a file or folder"""
        full_path = os.path.join(self.base_path, file_path)

        if not os.path.exists(full_path):
            return False

        if archive_name is None:
            archive_name = file_path + ".zip"

        archive_path = os.path.join(self.base_path, archive_name)

        if os.path.isdir(full_path):
            shutil.make_archive(archive_path.replace(".zip", ""), "zip", full_path)
        else:
            import zipfile

            with zipfile.ZipFile(archive_path, "w") as zipf:
                zipf.write(full_path, os.path.basename(full_path))

        print(f"Compressed {file_path} to {archive_name}")
        return True

    def list_directory(self, dir_path=""):
        """List contents of a directory"""
        full_path = os.path.join(self.base_path, dir_path)

        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            return []

        items = os.listdir(full_path)
        result = []

        for item in items:
            item_path = os.path.join(full_path, item)
            result.append(
                {
                    "name": item,
                    "is_dir": os.path.isdir(item_path),
                    "size": (
                        os.path.getsize(item_path) if os.path.isfile(item_path) else 0
                    ),
                }
            )

        return result
