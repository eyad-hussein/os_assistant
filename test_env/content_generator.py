import random
import json
import datetime
import lorem


class ContentGenerator:
    def __init__(self):
        self.python_snippets = [
            "def hello_world():\n    print('Hello, World!')\n\nhello_world()",
            "import random\n\nnum = random.randint(1, 100)\nprint(f'Random number: {num}')",
            "class Person:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n\n    def greet(self):\n        return f'Hello, my name is {self.name}!'\n\nperson = Person('Alice', 30)\nprint(person.greet())",
        ]

        self.javascript_snippets = [
            "function helloWorld() {\n  console.log('Hello, World!');\n}\n\nhelloWorld();",
            "const numbers = [1, 2, 3, 4, 5];\nconst doubled = numbers.map(num => num * 2);\nconsole.log(doubled);",
            "class Counter {\n  constructor() {\n    this.count = 0;\n  }\n  \n  increment() {\n    this.count++;\n  }\n}\n\nconst counter = new Counter();\ncounter.increment();\nconsole.log(counter.count);",
        ]

        self.markdown_snippets = [
            "# Project Title\n\n## Overview\nThis is a sample project.\n\n## Features\n- Feature 1\n- Feature 2\n- Feature 3",
            "# Meeting Notes\n\n## Attendees\n- John Doe\n- Jane Smith\n\n## Action Items\n1. Complete task A\n2. Review document B\n3. Schedule follow-up meeting",
            "# Tutorial\n\n## Step 1\nInstall dependencies.\n\n## Step 2\nConfigure settings.\n\n## Step 3\nRun the application.",
        ]

    def generate_python_file(self):
        """Generate content for a Python file"""
        return random.choice(self.python_snippets)

    def generate_javascript_file(self):
        """Generate content for a JavaScript file"""
        return random.choice(self.javascript_snippets)

    def generate_text_file(self, paragraphs=3):
        """Generate random text content"""
        return lorem.text()

    def generate_json_file(self):
        """Generate content for a JSON file"""
        data = {
            "id": random.randint(1000, 9999),
            "name": random.choice(
                ["Project Alpha", "Task Manager", "Data Analyzer", "System Monitor"]
            ),
            "version": f"{random.randint(0, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tags": random.sample(
                ["development", "testing", "production", "analysis", "research"],
                random.randint(1, 3),
            ),
            "settings": {
                "debug": random.choice([True, False]),
                "timeout": random.randint(30, 120),
                "retries": random.randint(1, 5),
            },
        }
        return json.dumps(data, indent=2)

    def generate_markdown_file(self):
        """Generate content for a Markdown file"""
        return random.choice(self.markdown_snippets)

    def generate_csv_file(self, rows=10):
        """Generate content for a CSV file"""
        headers = ["ID", "Name", "Date", "Value"]

        content = ",".join(headers) + "\n"

        for i in range(rows):
            row = [
                str(i + 1),
                f"Item-{random.randint(100, 999)}",
                (
                    datetime.datetime.now()
                    - datetime.timedelta(days=random.randint(0, 30))
                ).strftime("%Y-%m-%d"),
                str(random.randint(10, 1000)),
            ]
            content += ",".join(row) + "\n"

        return content

    def generate_log_file(self, entries=20):
        """Generate content for a log file"""
        log_levels = ["INFO", "DEBUG", "WARNING", "ERROR"]
        components = ["System", "Database", "Network", "UI", "API"]
        messages = [
            "Operation completed successfully",
            "Connection established",
            "Resource not found",
            "Invalid input detected",
            "Process started",
            "Process completed",
            "Timeout occurred",
            "Authentication failed",
            "Data validation error",
            "Cache miss",
        ]

        content = ""

        for _ in range(entries):
            timestamp = (
                datetime.datetime.now()
                - datetime.timedelta(minutes=random.randint(0, 1000))
            ).strftime("%Y-%m-%d %H:%M:%S")

            level = random.choice(log_levels)
            component = random.choice(components)
            message = random.choice(messages)

            content += f"[{timestamp}] [{level}] [{component}] {message}\n"

        return content

    def generate_file_content(self, file_extension):
        """Generate content based on file extension"""
        if file_extension == ".py":
            return self.generate_python_file()
        elif file_extension in [".js", ".jsx"]:
            return self.generate_javascript_file()
        elif file_extension == ".txt":
            return self.generate_text_file()
        elif file_extension == ".json":
            return self.generate_json_file()
        elif file_extension in [".md", ".markdown"]:
            return self.generate_markdown_file()
        elif file_extension == ".csv":
            return self.generate_csv_file()
        elif file_extension == ".log":
            return self.generate_log_file()
        else:
            return self.generate_text_file(1)  # Default to simple text
