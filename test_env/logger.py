import os
import datetime
import json


class Logger:
    def __init__(self, log_dir, console_output=True):
        self.log_dir = log_dir
        self.console_output = console_output

        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log_file = os.path.join(
            log_dir,
            f"operations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        self.stats_file = os.path.join(log_dir, "operation_stats.json")

        # Initialize statistics
        self.stats = {
            "operations": {
                "create_folder": 0,
                "delete_folder": 0,
                "create_file": 0,
                "delete_file": 0,
                "rename_item": 0,
                "edit_file": 0,
                "compress_file": 0,
                "other": 0,
            },
            "extensions": {},
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_operations": 0,
        }

    def log_operation(self, operation, path, details=None):
        """Log a file system operation"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Update statistics
        self.stats["total_operations"] += 1

        # Categorize operation
        op_type = operation.lower().split()[0]
        if op_type in self.stats["operations"]:
            self.stats["operations"][op_type] += 1
        else:
            self.stats["operations"]["other"] += 1

        # Track file extensions
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1]
            if ext:
                if ext in self.stats["extensions"]:
                    self.stats["extensions"][ext] += 1
                else:
                    self.stats["extensions"][ext] = 1

        # Format log entry
        log_entry = f"[{timestamp}] {operation}: {path}"
        if details:
            log_entry += f" - {details}"

        # Write to log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        # Output to console if enabled
        if self.console_output:
            print(log_entry)

        # Update stats file
        self._update_stats()

    def _update_stats(self):
        """Update the statistics file"""
        self.stats["last_updated"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

    def get_operation_history(self, limit=50):
        """Get recent operation history"""
        if not os.path.exists(self.log_file):
            return []

        with open(self.log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        return lines[-limit:] if limit > 0 else lines
