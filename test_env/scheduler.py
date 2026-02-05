import random
import time
import threading
import os
import datetime


class OperationScheduler:
    def __init__(self, file_ops, content_gen, min_interval=0, max_interval=300):
        self.file_ops = file_ops
        self.content_gen = content_gen
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.running = False
        self.thread = None

        # File extensions to use in random operations
        self.extensions = [
            ".txt",
            ".py",
            ".js",
            ".json",
            ".md",
            ".csv",
            ".log",
            ".html",
            ".css",
            ".yaml",
            ".xml",
            ".ini",
            ".conf",
        ]

        # Weights for different operations (higher = more frequent)
        self.operation_weights = {
            "create_folder": 15,
            "create_file": 25,
            "edit_file": 30,
            "rename_item": 10,
            "delete_file": 10,
            "delete_folder": 5,
            "compress_file": 5,
        }

        # Initialize counters
        self.operation_count = 0
        self.folders_created = set()

        # Add random name components for more natural file/folder names
        self.adjectives = [
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "black",
            "white",
            "big",
            "small",
            "fast",
            "slow",
            "smart",
            "clever",
            "happy",
            "sad",
            "busy",
            "lazy",
            "shiny",
            "dull",
            "sharp",
            "flat",
            "round",
        ]

        self.nouns = [
            "dog",
            "cat",
            "bird",
            "fish",
            "horse",
            "car",
            "bike",
            "boat",
            "plane",
            "train",
            "house",
            "tree",
            "flower",
            "river",
            "mountain",
            "ocean",
            "desert",
            "forest",
            "city",
            "village",
            "project",
            "document",
            "report",
            "data",
            "image",
            "video",
            "audio",
            "code",
            "script",
            "note",
            "plan",
            "idea",
            "concept",
            "design",
            "model",
            "system",
            "app",
            "game",
        ]

        self.tech_terms = [
            "api",
            "function",
            "class",
            "module",
            "library",
            "framework",
            "server",
            "client",
            "database",
            "cache",
            "queue",
            "stack",
            "heap",
            "algorithm",
            "interface",
            "component",
            "service",
            "controller",
            "view",
            "model",
            "config",
            "settings",
            "profile",
            "account",
            "user",
            "admin",
            "backup",
            "archive",
            "log",
            "debug",
            "test",
            "prod",
            "dev",
            "staging",
            "release",
        ]

    def start(self):
        """Start the scheduler"""
        if self.running:
            return False

        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        return True

    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return True

    def _run(self):
        """Main scheduler loop"""
        while self.running:
            # Perform a random operation
            self._perform_random_operation()

            # Sleep for a random interval
            interval = random.randint(self.min_interval, self.max_interval)
            print(f"Sleeping for {interval} seconds")

            # Sleep in small increments to allow for clean shutdown
            for _ in range(interval):
                if not self.running:
                    break
                time.sleep(1)

    def _perform_random_operation(self):
        """Perform a random file system operation"""
        # Choose operation based on weights
        operations = list(self.operation_weights.keys())
        weights = list(self.operation_weights.values())
        operation = random.choices(operations, weights=weights, k=1)[0]

        # Increment counter
        self.operation_count += 1
        print(f"Operation #{self.operation_count}: {operation}")

        # Perform the selected operation
        if operation == "create_folder":
            self._create_random_folder()
        elif operation == "create_file":
            self._create_random_file()
        elif operation == "edit_file":
            self._edit_random_file()
        elif operation == "rename_item":
            self._rename_random_item()
        elif operation == "delete_file":
            self._delete_random_file()
        elif operation == "delete_folder":
            self._delete_random_folder()
        elif operation == "compress_file":
            self._compress_random_item()

    def _generate_random_name(self, use_tech=False):
        """Generate a random natural-sounding name"""
        name_parts = []

        # 50% chance to add an adjective
        if random.random() < 0.6:
            name_parts.append(random.choice(self.adjectives))

        # Always add a noun
        name_parts.append(random.choice(self.nouns))

        # 30% chance to add a tech term if requested
        if use_tech and random.random() < 0.6:
            name_parts.append(random.choice(self.tech_terms))

        # 20% chance to add a number
        if random.random() < 0.5:
            name_parts.append(str(random.randint(1, 999)))

        # Join parts with underscores or hyphens
        separator = random.choice(["_", "-"]) if random.random() < 0.7 else ""
        name = separator.join(name_parts)

        return name

    def _create_random_folder(self):
        """Create a random folder with a natural-sounding name"""
        folder_name = self._generate_random_name(use_tech=True)

        # Sometimes create nested folders
        if self.folders_created and random.random() < 0.3:
            parent = random.choice(list(self.folders_created))
            folder_path = os.path.join(parent, folder_name)
        else:
            folder_path = folder_name

        result = self.file_ops.create_folder(folder_path)
        if result:
            self.folders_created.add(folder_path)

    def _create_random_file(self):
        """Create a random file with a natural-sounding name"""
        ext = random.choice(self.extensions)
        file_name = f"{self._generate_random_name()}{ext}"

        # Sometimes place in subfolders
        if self.folders_created and random.random() < 0.7:
            parent = random.choice(list(self.folders_created))
            file_path = os.path.join(parent, file_name)
        else:
            file_path = file_name

        content = self.content_gen.generate_file_content(ext)
        self.file_ops.create_file(file_path, content)

    def _edit_random_file(self):
        """Edit a random existing file"""
        files = self._get_existing_files()
        if not files:
            return

        file_path = random.choice(files)
        ext = os.path.splitext(file_path)[1]

        # Sometimes append, sometimes overwrite
        append = random.random() < 0.3
        content = self.content_gen.generate_file_content(ext)

        if append:
            content = (
                "\n\n# Updated "
                + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + "\n"
                + content
            )

        self.file_ops.edit_file(file_path, content, append)

    def _rename_random_item(self):
        """Rename a random file or folder with a natural-sounding name"""
        items = self._get_existing_items()
        if not items:
            return

        old_path = random.choice(items)

        # Determine if it's a file or folder
        is_file = os.path.splitext(old_path)[1] != ""

        if is_file:
            base, ext = os.path.splitext(old_path)
            dir_name = os.path.dirname(base)
            new_name = f"{self._generate_random_name()}{ext}"
            if dir_name:
                new_path = os.path.join(dir_name, new_name)
            else:
                new_path = new_name
        else:
            dir_name = os.path.dirname(old_path)
            new_name = self._generate_random_name(use_tech=True)
            if dir_name:
                new_path = os.path.join(dir_name, new_name)
            else:
                new_path = new_name

        result = self.file_ops.rename_item(old_path, new_path)

        # Update folder tracking
        if result and old_path in self.folders_created:
            self.folders_created.remove(old_path)
            self.folders_created.add(new_path)

    def _delete_random_file(self):
        """Delete a random file"""
        files = self._get_existing_files()
        if not files:
            return

        file_path = random.choice(files)
        self.file_ops.delete_file(file_path)

    def _delete_random_folder(self):
        """Delete a random folder"""
        if not self.folders_created:
            return

        folder_path = random.choice(list(self.folders_created))
        result = self.file_ops.delete_folder(folder_path)

        if result:
            # Remove this folder and any subfolders from tracking
            folders_to_remove = set()
            for folder in self.folders_created:
                if folder == folder_path or folder.startswith(folder_path + os.sep):
                    folders_to_remove.add(folder)

            self.folders_created -= folders_to_remove

    def _compress_random_item(self):
        """Compress a random file or folder"""
        items = self._get_existing_items()
        if not items:
            return

        item_path = random.choice(items)
        self.file_ops.compress_file(item_path)

    def _get_existing_files(self):
        """Get a list of existing files"""
        result = []
        base_path = self.file_ops.base_path

        for root, _, files in os.walk(base_path):
            rel_root = os.path.relpath(root, base_path)
            if rel_root == ".":
                rel_root = ""

            for file in files:
                if file.endswith(".zip") or file.endswith(".log"):
                    continue  # Skip zip and log files
                file_path = os.path.join(rel_root, file)
                result.append(file_path)

        return result

    def _get_existing_items(self):
        """Get a list of existing files and folders"""
        files = self._get_existing_files()
        folders = list(self.folders_created)
        return files + folders
