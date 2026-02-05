import random
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from os_assistant.utils.settings import MODEL_TYPE
from os_assistant.utils.model_factory import create_model
from tracer.config import LogDomain

from os_assistant.tools.agentic_rag.application.search import search_logs
from os_assistant.tools.code_agent.wrapper import code_execute_tool
from os_assistant.utils.settings import MODEL_BASE_URL
import os

TEST_ENV_ABS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'test_env'))
TEST_ENV_POSIX = TEST_ENV_ABS.replace('\\', '/')

from ..config.config import (
    DATASET_LLM_MODEL,
    MAX_QUESTIONS_PER_LOG,
    MIN_QUESTIONS_PER_LOG,
)


class QuestionGenerator:
    """Generates structured questions for dataset creation with focus on file system operations"""

    def __init__(
        self, model_name: str = DATASET_LLM_MODEL, base_url: str = MODEL_BASE_URL
    ):
        """Initialize with model parameters"""
        # Use Ollama model only when configured; otherwise fall back to configured provider
        if MODEL_TYPE and MODEL_TYPE.upper() == "OLLAMA" and base_url:
            self.llm = ChatOllama(model=model_name, temperature=0.7, base_url=base_url)
            self.code_llm = ChatOllama(model=model_name, temperature=0.2, base_url=base_url)
        else:
            # create_model will use settings to instantiate the correct provider
            self.llm = create_model(model=model_name, timeout=30)
            self.code_llm = create_model(model=model_name, timeout=30)

    def generate_questions_from_logs(
        self,
        logs: list[dict[str, Any]],
        num_questions: int | None = None,
        domain_hint: str | None = "file_system",
        previous_questions: list[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Generate structured questions based on log content with explicit type and expected response.

        Args:
            logs: List of log dictionaries
            num_questions: Optional number of questions to generate (random if None)
            domain_hint: Optional domain hint to focus question generation
            previous_questions: Optional list of recent questions to avoid duplicating

        Returns:
            List of dictionaries with structured questions and metadata
        """
        if not logs:
            return []

        # If num_questions not specified, choose a random number within bounds
        if num_questions is None:
            num_questions = random.randint(MIN_QUESTIONS_PER_LOG, MAX_QUESTIONS_PER_LOG)

        # Format the logs for the prompt
        formatted_logs = ""
        for i, log in enumerate(logs):
            formatted_logs += f"Log {i + 1} (Timestamp: {log['timestamp']}):\n"
            formatted_logs += f"{log['log_text']}\n\n"

        # System prompt for structured question generation
        system_prompt = f"""You are an expert at generating realistic, precise, and diverse Linux file system questions from system logs.
Given activity logs showing interactions with the directory {TEST_ENV_ABS} and its contents, write a series of user questions that could reasonably arise from reviewing those logs.

Each question must follow this exact format:
---
question: [The user's question here - make it natural and conversational]
type: [command OR information]
expected_response: [Detailed command with options OR comprehensive explanation]
---

Focus all questions on the path "{TEST_ENV_ABS}" and its contents.

Guidelines for creating highly relevant and diverse questions:
1- Every question must be grounded in actions from the logs, like file creation, editing, moving, or reading within {TEST_ENV_ABS} or its subdirectories.
2- Use specific file or folder names observed in the logs (e.g., data, scripts, results.csv, etc.).
3- Use both types:"command" for questions seeking Linux terminal commands and "information" for questions seeking explanations of Linux behavior or concepts
For “command” questions, include:
Viewing the first or last 100 characters/lines of a file

Checking modification timestamps

Comparing file sizes or searching for similarly sized files

Viewing or counting files with specific extensions

Recursive actions like listing nested directories

For “information” questions, include:
How symbolic links or file timestamps work

Why certain files change in size or timestamp

Differences between hidden files and regular files in this directory

Behavior of tools like diff, find, or stat in the context of the directory
Your questions MUST be directly derived from the logs, such as:
- If logs show operations on files in "{TEST_ENV_ABS}/data", ask about those specific files
- If logs show creation of new directories, ask about making or listing directories
- If logs show file modification times, ask about checking or monitoring file changes

Examples of good questions:
---
question: How can I view just the first 100 characters from the file {TEST_ENV_ABS}/data/raw.txt?
type: command
expected_response: head -c 100 "{TEST_ENV_POSIX}/data/raw.txt"
---

"""

        # Check if we have previous questions to avoid
        recent_examples = ""
        if previous_questions and len(previous_questions) > 0:
            recent_examples = (
                "RECENTLY GENERATED QUESTIONS (AVOID CREATING SIMILAR ONES):\n"
            )
            for i, q in enumerate(previous_questions[-5:]):  # Take up to 5 most recent
                recent_examples += f"{i + 1}. {q}\n"
            recent_examples += "\n"

        # Human prompt with logs and specific instructions
        human_prompt = f"""Here are Linux system logs to generate questions from:

{formatted_logs}

{recent_examples}Please generate {num_questions} realistic file system questions based on THESE SPECIFIC LOGS.

IMPORTANT REQUIREMENTS:
1. All questions MUST be directly related to the activities shown in these logs
2. Reference specific files, paths, commands, and actions mentioned in the logs
3. ALWAYS focus on the "{TEST_ENV_ABS}" directory and its contents
4. Include a mix of "command" and "information" type questions
5. For command questions, the expected_response MUST include the full command with options
6. For information questions, the expected_response MUST be a comprehensive explanation
7. Each question MUST be separated with a blank line

Examples of good questions based on sample logs:
- If logs show "created {TEST_ENV_ABS}/data/temp", ask "How can I list all files in the newly created temp directory?"
- If logs show "modified {TEST_ENV_ABS}/config.ini", ask "How can I monitor changes to the config.ini file in real-time?"
- If logs show "deleted {TEST_ENV_ABS}/logs/old_data", ask "What command would restore the deleted old_data directory if it was backed up?"
"""

        # Generate questions
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        print(f"Generating questions from {len(logs)} logs...")
        response = self.llm.invoke(messages)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Parse the structured questions
        parsed_questions = self._parse_structured_questions(response_text)

        # Create result with metadata
        result = []
        for question_data in parsed_questions[:num_questions]:
            result.append(
                {
                    "question": question_data["question"],
                    "type": question_data["type"],
                    "expected_response": question_data["expected_response"],
                    "source_logs": [log["log_number"] for log in logs],
                    "domain": domain_hint or logs[0].get("domain", "file_system"),
                    "timestamps": [log["timestamp"] for log in logs],
                    "generated_type": "log_based",  # Add the generated_type field
                }
            )

        return result

    def enhance_with_rag(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Enhance question answers using RAG search.

        Args:
            questions: List of questions to enhance

        Returns:
            Enhanced questions with better answers from RAG
        """
        enhanced_questions = []

        for question in questions:
            original_question = question.get("question", "")
            if not original_question:
                enhanced_questions.append(question)
                continue

            print(f"Enhancing answer for question: {original_question}")

            try:
                # Determine which domain to search based on question metadata
                domain_str = question.get("domain", "file_system")
                domain = LogDomain.FS  # Default to file system

                # Convert string domain to enum if needed
                if domain_str == "users":
                    domain = LogDomain.USERS
                elif domain_str == "packages":
                    domain = LogDomain.PKG
                elif domain_str == "networking":
                    domain = LogDomain.NET

                # Search logs related to the question
                logs, summaries = search_logs(
                    query=original_question,
                    domains=[domain],
                    top_k=3,
                    summarize=True,
                    auto_init=True,
                )

                # Skip enhancement if no relevant logs found
                if not logs:
                    print("No relevant logs found for enhancing answer")
                    enhanced_questions.append(question)
                    continue

                # Format the RAG results
                rag_context = ""
                for i, log in enumerate(logs):
                    rag_context += (
                        f"Log #{log['log_number']} (Timestamp: {log['timestamp']})\n"
                    )

                    # Include summary if available
                    if summaries and i < len(summaries):
                        rag_context += f"Summary: {summaries[i]}\n"

                    # Add the log text
                    rag_context += f"Content: {log['log_text']}\n\n"

                # Generate enhanced answer based on RAG results
                question_type = question.get("type", "information")
                original_response = question.get("expected_response", "")

                prompt = f"""Based on this question and additional context from the user's system logs, 
                create an improved response that incorporates specific details from the logs.

                Question: {original_question}
                
                Question type: {question_type}
                
                Original response: {original_response}
                
                Additional context from system logs:
                {rag_context}
                
                Create a comprehensive, specific response that:
                1. Directly addresses the question
                2. Incorporates relevant details from the system logs
                3. Is personalized to the user's actual system
                4. {"Includes the exact command with proper options" if question_type == "command" else "Provides thorough explanation with examples"}
                
                Your response should be more specific and helpful than the original response.
                """

                rag_response = self.llm.invoke([HumanMessage(content=prompt)])
                enhanced_answer = (
                    rag_response.content
                    if hasattr(rag_response, "content")
                    else str(rag_response)
                )

                # Update the question with enhanced answer
                question["expected_response"] = enhanced_answer
                question["rag_enhanced"] = True
                question["rag_logs"] = [log.get("log_number") for log in logs]

                print("Answer enhanced with RAG results")

            except Exception as e:
                print(f"[ERROR] Failed to enhance answer with RAG: {str(e)}")

            enhanced_questions.append(question)

        return enhanced_questions

    def generate_random_questions(
        self, num_questions: int = 5, domain: str = "file_system"
    ) -> list[dict[str, Any]]:
        """
        Generate random file system questions not tied to specific logs.

        Args:
            num_questions: Number of questions to generate
            domain: Domain to focus on (defaults to file_system)

        Returns:
            List of dictionaries with structured questions
        """
        system_prompt = """You are an expert at creating realistic and diverse Linux file system questions.
Generate questions that a user might ask without reference to specific logs.

IMPORTANT: Each question MUST follow this EXACT format:
---
question: [The user's question here - make it natural and conversational]
type: [command OR information]
expected_response: [Detailed command with options OR comprehensive explanation]
---


Create diverse, realistic questions that Linux users commonly ask, focusing on:
1. File operations (find, copy, move, delete) 
2. Permission issues and ownership for files
3. Directory structure and navigation
4. File content searching and manipulation
5. Disk space and storage management 

Include a variety of question formulations:
- "How do I check the size of files?"
- "Can you help me find all .log files?"
- "What's the best way to monitor changes?"
- "Is it possible to search for text within all files?"
"""

        human_prompt = f"""Please generate {num_questions} realistic file system questions that Linux users might ask.

IMPORTANT REQUIREMENTS:
1. All questions MUST follow the exact format specified
2. Include a mix of "command" and "information" type questions
3. For command questions, the expected_response MUST include the full command with options
4. For information questions, the expected_response MUST be a comprehensive explanation
5. Each question MUST be separated with a blank line
6. Make questions specific and practical, as if from real Linux users

Examples of good questions:
- "How do I find all files larger than 100MB?"
- "What's the command to count the number of lines in all text files?"
- "How can I monitor changes to file?"
- "Is it possible to encrypt sensitive files?"
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = self.llm.invoke(messages)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Parse the structured questions
        parsed_questions = self._parse_structured_questions(response_text)

        # Create result with metadata
        result = []
        for question_data in parsed_questions[:num_questions]:
            result.append(
                {
                    "question": question_data["question"],
                    "type": question_data["type"],
                    "expected_response": question_data["expected_response"],
                    "source_logs": [],  # No source logs for random questions
                    "domain": domain,
                    "generated_type": "random",  # Add the generated_type field
                }
            )

        return result

    # TODO: I Think we should add a tree of the folders and subfolders so the agent can ask good questions!
    """
    He doesn't know what inside that directory , either we should 
    give it to him , or telling it in the prompt to generate what inside 
    the directory first in the code (overhead), i recommend the first.
    """

    def generate_code_execution_questions(
        self, num_questions: int = 5
    ) -> list[dict[str, Any]]:
        """
        Generate questions specifically designed to trigger the code execution agent.
        These questions focus on file system analysis that requires running code.

        Args:
            num_questions: Number of questions to generate

        Returns:
            List of dictionaries with structured questions
        """
        # Directory tree to provide context for questions
        directory_tree = f"""{TEST_ENV_ABS}/
├── __pycache__/
│   ├── content_generator.cpython-310.pyc
│   ├── content_generator.cpython-311.pyc
│   ├── file_operations.cpython-310.pyc
│   ├── file_operations.cpython-311.pyc
│   └── scheduler.cpython-311.pyc
├── data/
│   ├── big_plan_827/
│   ├── bike-240/
│   ├── bike_client/
│   ├── bird/
│   │   ├── flatimage/
│   │   └── round-fish-algorithm-383/
│   ├── birdcache50/
│   ├── black-data-log/
│   ├── black_project_framework_156/
│   │   ├── flat-video-profile/
│   │   │   └── bike/
│   │   │       └── round-mountain-698.txt
│   │   ├── image.txt
│   │   ├── model-481.zip
│   │   └── slow_system.html
│   ├── blackbird540/
│   ├── blue-data/
│   ├── blue_concept_function_170/
│   │   ├── bigboat767.html
│   │   ├── lazy-river.ini
│   │   └── white-idea-975.log
│   ├── blue_ocean/
│   │   ├── round_fish.yaml
│   │   ├── smart_flower_601.html
│   │   ├── smart_flower_601.html.zip
│   │   └── tree.css
│   ├── boat/
│   │   ├── dullmodel/
│   │   │   └── sad_app_546.log
│   │   └── dullmodel.zip
│   ├── boat-release-539/
│   ├── busy_model/
│   │   └── plane637.log
│   ├── busy_script/
│   │   ├── dog.html
│   │   ├── dullapp754.log
│   │   ├── horse206.xml
│   │   ├── purpleriver.css
│   │   └── red_data.yaml
│   ├── car/
│   │   └── red_data.py
│   ├── clever-fish-database-144/
│   │   ├── clevermountain.txt
│   │   └── shinytree105.py
│   ├── clever-idea-config-883/
│   ├── concept-stack-263/
│   ├── concept_admin_220/
│   ├── data_debug_48/
│   ├── desert101/
│   │   └── sharp-forest-backup-291/
│   │       └── lazy_audio_368.xml
│   ├── design_controller/
│   │   └── dull_forest_client_804/
│   ├── document-class-786/
│   ├── document_database_700/
│   │   └── clever-document.py
│   ├── dull-game-364/
│   ├── dull_cat_user_139/
│   ├── dullvideo590/
│   │   └── code43/
│   ├── fast_script_config/
│   │   └── small_train.css
│   ├── fastimage/
│   ├── flat_system_999/
│   │   └── plane_521.conf
│   ├── flathorsebackup282/
│   │   └── lazy-boat-963.js
│   ├── flatnoteinterface319/
│   ├── flower_dev/
│   ├── green-flower-backup/
│   │   ├── ocean.yaml
│   │   └── ocean.yaml.zip
│   ├── greentreedebug521/
│   ├── happy-boat/
│   ├── happy-cat-admin/
│   │   ├── small_script_567/
│   │   │   ├── design_469.md
│   │   │   └── design_469.md.zip
│   │   ├── app.py
│   │   ├── house.log
│   │   ├── ocean719.conf
│   │   └── report.log
│   ├── happy-document-300/
│   ├── happy-script-backup/
│   ├── happy_car_backup/
│   │   └── system.xml
│   ├── happy_dog_service/
│   │   └── busy_document_algorithm/
│   ├── happy_model/
│   │   ├── blue-car-stack-349/
│   │   ├── scriptcomponent/
│   │   └── forest.json
│   ├── happy_plane_archive/
│   │   └── horse194/
│   ├── house-queue/
│   │   └── city.py
│   ├── idea/
│   │   └── fish192.html.zip
│   ├── idea_framework_87/
│   ├── idea_stack_219/
│   │   ├── audio/
│   │   └── shinyhouse947.csv
│   ├── image/
│   ├── image-user/
│   │   ├── river_dev/
│   │   │   └── mountain.py.zip
│   │   ├── small-document-interface/
│   │   ├── bike.yaml
│   │   ├── project.html.zip
│   │   └── sharp_flower_792.js
│   ├── lazy-plane/
│   │   └── model.ini
│   ├── model-settings/
│   ├── model_backup_227/
│   │   ├── clever_fish.md
│   │   └── clever_fish.md.zip
│   ├── mountain-database-919/
│   ├── note957/
│   ├── plane/
│   │   └── app-855.log
│   ├── plane458/
│   ├── projects/
│   ├── purple-fish/
│   ├── purple-script/
│   │   ├── sharp-forest-backup-291/
│   │   ├── lazy-tree-733.xml
│   │   └── ocean.csv
│   ├── purple_bird_interface_117/
│   │   ├── forest.json
│   │   ├── plan.json
│   │   └── yellowvillage.log
│   ├── purple_note_325/
│   │   └── fish-217.log
│   ├── purplehorse/
│   ├── redcity359/
│   │   └── small_concept_77/
│   │       ├── purple-design/
│   │       │   ├── dog886/
│   │       │   ├── smallcity/
│   │       │   │   └── report_252.py
│   │       │   └── small-flower.py
│   │       ├── document517.txt
│   │       └── purple-design.zip
│   ├── redplanqueue311/
│   ├── river_algorithm/
│   ├── rivercontroller/
│   │   └── scriptcomponent/
│   │       └── whitescript.xml
│   ├── roundcodecontroller511/
│   │   └── train.log
│   ├── roundocean472/
│   │   ├── script-130/
│   │   │   ├── clever_audio_profile/
│   │   │   ├── projectlog359/
│   │   │   │   └── small-game.js
│   │   │   ├── round_data/
│   │   │   │   └── sharp_cat_library/
│   │   │   ├── data-382.ini
│   │   │   ├── purple-train.md
│   │   │   ├── shiny-video.json
│   │   │   └── trainstack155.zip
│   │   ├── shinydesign/
│   │   │   ├── clever_audio_profile/
│   │   │   │   └── white-dog-account/
│   │   │   │       └── black_concept_155/
│   │   │   │           └── image_730.log
│   │   │   ├── project/
│   │   │   │   └── shiny_code.json
│   │   │   └── trainstack155/
│   │   │       └── sharp_cat_library/
│   │   │           └── fish_staging/
│   │   ├── white-app-307/
│   │   │   ├── dog/
│   │   │   ├── flower/
│   │   │   ├── design-427.json
│   │   │   └── purple_design.txt.zip
│   │   ├── red_idea_385.txt
│   │   └── sharpproject.log
│   ├── sad-car/
│   ├── sad-image-module-169/
│   │   ├── dullappconfig970/
│   │   └── blue-car-496.ini
│   ├── sad_game_834/
│   ├── script_class/
│   ├── sharp-project-admin/
│   │   ├── sad-note.conf
│   │   └── script.py
│   ├── sharp-train-algorithm/
│   ├── sharpdocumentclient881/
│   ├── shiny_desert_config/
│   ├── slow_image_954/
│   ├── slow_river_function_343/
│   ├── slowtrainuser632/
│   ├── small-system-log-637/
│   ├── small-village-view-231/
│   │   └── sad_train.css
│   ├── small_project_user/
│   │   └── smartapp353.log
│   ├── smart_report_interface_503/
│   ├── smartdesertfunction/
│   │   └── round-flower.py
│   ├── smarttrainservice/
│   │   └── dullmodel850.conf
│   ├── system/
│   │   ├── plan.xml
│   │   ├── purple_image_645.yaml
│   │   └── white_house.json
│   ├── village_prod/
│   │   └── slow_plane_735.txt
│   ├── white-mountain-admin/
│   ├── white-script-framework-760/
│   ├── whitedog/
│   │   ├── blackvideocontroller/
│   │   ├── yellow-bike-library/
│   │   └── video798.html
│   ├── yellow_note_316/
│   ├── yellow_report_api/
│   ├── yellow_village_150/
│   ├── bird.log
│   ├── bird.zip
│   ├── black-audio-768.conf
│   ├── blue-app.conf
│   ├── boat_206.log
│   ├── busy-train.csv
│   ├── busy_dog.yaml
│   ├── busyvideo225.zip
│   ├── cat.ini
│   ├── cat.js
│   ├── data_debug_48.zip
│   ├── desert-421.ini
│   ├── desert-model.zip
│   ├── desert.log
│   ├── desert101.zip
│   ├── design_controller.zip
│   ├── document_249.html
│   ├── document_database_700.zip
│   ├── dull-game.log
│   ├── fast-app-108.xml
│   ├── fast_flower_515.zip
│   ├── fish735.yaml
│   ├── flathorse.log
│   ├── green_house_350.ini
│   ├── happy-boat.zip
│   ├── happy-cat-admin.zip
│   ├── happy-report-service-544.zip
│   ├── image.js
│   ├── image.zip
│   ├── mountain_501.yaml
│   ├── note-693.log
│   ├── note_78.js
│   ├── ocean_497.csv
│   ├── personal_data.zip
│   ├── plane.py.zip
│   ├── purple-cat-891.log
│   ├── purple-design.md
│   ├── purple-design.md.zip
│   ├── purple-note.ini.zip
│   ├── purple-project.js
│   ├── redcity359.zip
│   ├── report-480.conf.zip
│   ├── round-script.css.zip
│   ├── saddog976.csv
│   ├── sharp-train-algorithm.zip
│   ├── shiny-river.py
│   ├── shiny-river.py.zip
│   ├── shiny_city.yaml
│   ├── shiny_house.csv
│   ├── slow-project-332.log
│   ├── slow_dog_954.css
│   ├── slow_dog_framework_986.zip
│   ├── small-mountain-179.css
│   ├── small-river.zip
│   ├── smallreport.csv
│   ├── smallreport.csv.zip
│   ├── smart-horse-594.py
│   ├── smart_cat_497.txt
│   ├── smart_report_interface_503.zip
│   ├── system670.yaml
│   ├── white-script-framework-760.zip
│   └── whiteplan.json
├── README.md
├── content_generator.py
├── directory_viewer.py
├── file_operations.py
├── logger.py
├── main.py
└── scheduler.py
"""

        system_prompt = f"""You are an expert at creating Linux questions that require code execution.
Generate questions about file system analysis that would require running Python or shell code to answer.

IMPORTANT: Each question MUST follow this EXACT format:
---
question: [The user's question here about file system analysis]
type: command
expected_response: [The code that would need to be executed plus explanation]
code_solution: [Python or shell code that would solve this]
---

CRUCIAL PATH INFORMATION:
- The FULL absolute path to the test environment is: "{TEST_ENV_ABS}"
- Every question MUST include this FULL PATH in the question text itself
- ALL code solutions MUST use this EXACT path when accessing files or directories
- The working directory for code execution might be different, so ALWAYS use absolute paths
- In Python code, use double backslashes (\\\\) or forward slashes (/) for Windows paths
- In shell code for Windows, also use double backslashes or forward slashes

Here is the actual directory structure you should reference in your questions:
{directory_tree}

Focus on questions that require file system analysis:
1. Finding largest/smallest files or directories in {TEST_ENV_ABS}
2. Analyzing file types and distributions (like how many .zip, .log, .py files exist)
3. Identifying duplicate files (like the .zip files in various directories)
4. Finding recently modified files across the directory structure
5. Analyzing disk usage patterns for specific subdirectories
6. Searching for files with specific content
7. Comparing subdirectories (e.g., which has more .log files)
8. Finding the deepest nested directories
9. Identifying empty directories
10. Analyzing file naming patterns

INSTRUCTIONS FOR CODE SOLUTIONS:
- For Python solutions, use libraries like os, pathlib, glob, or subprocess
- Always import required libraries at the beginning of your code
- For shell solutions on Windows, ensure commands are compatible with Windows CMD or PowerShell
- For file content analysis, include proper error handling
- Ensure code is complete and runnable as-is (no pseudocode)
- ALWAYS use ABSOLUTE paths in all code, never relative paths

Reference SPECIFIC files and directories from the tree in your questions. For example:
- "What are the contents of {TEST_ENV_ABS}/data/happy-cat-admin/report.log?"
- "How can I find all .zip files larger than 1MB in {TEST_ENV_ABS}/data?"
- "What's the distribution of file types in {TEST_ENV_ABS}/data/blue_ocean compared to {TEST_ENV_ABS}/data/purple-script?"
"""

        human_prompt = f"""Please generate {num_questions} questions about file system analysis that would require code execution.
Use the provided directory structure to make your questions specific and realistic.

IMPORTANT REQUIREMENTS:
1. All questions MUST follow the exact format specified
2. Questions MUST reference SPECIFIC files and directories that exist in the provided tree structure
3. EVERY question MUST include the FULL PATH "{TEST_ENV_ABS}" in the question text
4. Each code solution MUST use the FULL ABSOLUTE PATH in all file operations
5. Questions should require analysis that's easiest with Python or complex shell scripts
6. Include the code_solution field with working Python or shell code
7. Each question MUST be separated with a blank line
8. Make sure the code actually works if someone were to run it - include all necessary imports and error handling

Examples of good questions based on the actual directory structure:
- "What are the 5 largest .zip files in {TEST_ENV_ABS}/data and their sizes?"
- "How many log files are in {TEST_ENV_ABS}/data/happy-cat-admin and what's their total size?"
- "Can you analyze the distribution of file types in {TEST_ENV_ABS}/data/blue_ocean compared to {TEST_ENV_ABS}/data/purple-script?"
- "What's the content of {TEST_ENV_ABS}/data/black_project_framework_156/flat-video-profile/bike/round-mountain-698.txt? Can you analyze it for common words?"
- "Which subdirectory in {TEST_ENV_ABS}/data contains the most nested structure? How deep does it go?"
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        response = self.llm.invoke(messages)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Parse the structured questions
        parsed_questions = self._parse_structured_questions(
            response_text, include_code=True
        )

        # For each question, actually run the code execution tool to get a real answer
        result = []
        for question_data in parsed_questions[:num_questions]:
            question_text = question_data["question"]

            try:
                # Use the code_execute_tool to get a real response
                tool_result = code_execute_tool(question_text)

                # Add the real execution results to the question data
                question_data["actual_code"] = tool_result.get("code", "")
                question_data["execution_result"] = tool_result.get(
                    "execution_result", ""
                )
                question_data["agent_output"] = tool_result.get("agent_output", "")

                # Use the actual result to improve the expected response
                if tool_result.get("agent_output"):
                    # Generate a better expected response based on the code execution
                    improved_response_prompt = f"""Based on this file system question and the code execution results, 
                    create a comprehensive response that explains both the approach and the results:
                    
                    Question: {question_text}
                    
                    Code used: {tool_result.get("code", "")}
                    
                    Execution result: {tool_result.get("execution_result", "")}
                    
                    Agent analysis: {tool_result.get("agent_output", "")}
                    
                    Create a clear, helpful response that answers the original question completely.
                    """

                    improved_response = self.code_llm.invoke(
                        [HumanMessage(content=improved_response_prompt)]
                    )
                    improved_text = (
                        improved_response.content
                        if hasattr(improved_response, "content")
                        else str(improved_response)
                    )

                    # Update the expected response with the improved version
                    question_data["expected_response"] = improved_text

            except Exception as e:
                print(f"Error executing code for question '{question_text}': {str(e)}")

            result.append(
                {
                    "question": question_data["question"],
                    "type": question_data["type"],
                    "expected_response": question_data["expected_response"],
                    "source_logs": [],  # No source logs for code execution questions
                    "domain": "file_system",
                    "generated_type": "code_execution",  # Add the generated_type field
                    "code_solution": question_data.get("code_solution", ""),
                    "actual_code": question_data.get("actual_code", ""),
                    "execution_result": question_data.get("execution_result", ""),
                    "agent_output": question_data.get("agent_output", ""),
                }
            )

        return result

    def _parse_structured_questions(
        self, text: str, include_code: bool = False
    ) -> list[dict[str, Any]]:
        """
        Parse the structured questions from the response text.

        Args:
            text: Response text from the LLM
            include_code: Whether to look for code_solution field

        Returns:
            List of dictionaries with parsed question data
        """
        questions = []
        current_question = {}
        current_field = None

        # Split text into lines
        lines = text.split("\n")

        for line in lines:
            line = line.strip()

            # Skip empty lines unless we're in the middle of a field
            if (
                not line
                and current_field != "expected_response"
                and current_field != "code_solution"
            ):
                # If we have a complete question, add it
                if (
                    current_question
                    and "question" in current_question
                    and "type" in current_question
                    and "expected_response" in current_question
                ):
                    # Only add if it has the minimum required fields
                    questions.append(current_question)
                    current_question = {}
                current_field = None
                continue

            # Check for field markers
            if line.startswith("question:") or line.startswith("question :"):
                current_field = "question"
                value = line.split(":", 1)[1].strip()
                current_question[current_field] = value
            elif line.startswith("type:") or line.startswith("type :"):
                current_field = "type"
                value = line.split(":", 1)[1].strip().lower()
                # Ensure type is either command or information
                if value in ["command", "information"]:
                    current_question[current_field] = value
                else:
                    # Default to command if unrecognized
                    current_question[current_field] = "command"
            elif line.startswith("expected_response:") or line.startswith(
                "expected_response :"
            ):
                current_field = "expected_response"
                value = line.split(":", 1)[1].strip()
                current_question[current_field] = value
            elif include_code and (
                line.startswith("code_solution:") or line.startswith("code_solution :")
            ):
                current_field = "code_solution"
                value = line.split(":", 1)[1].strip()
                current_question[current_field] = value
            elif current_field:
                # Append to the current field's value
                current_question[current_field] = (
                    current_question.get(current_field, "") + " " + line
                )

        # Add the last question if it's complete
        if (
            current_question
            and "question" in current_question
            and "type" in current_question
            and "expected_response" in current_question
        ):
            questions.append(current_question)

        # Clean up the questions
        for q in questions:
            for key in q:
                if isinstance(q[key], str):
                    q[key] = q[key].strip()

        return questions
