import random
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from tracer.config import LogDomain

from os_assistant.config.settings import MODEL_BASE_URL
from os_assistant.tools.agentic_rag.application.search import search_logs
from os_assistant.tools.code_agent.wrapper import code_execute_tool

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
        self.llm = ChatOllama(model=model_name, temperature=0.7, base_url=base_url)
        self.code_llm = ChatOllama(model=model_name, temperature=0.2, base_url=base_url)

    def generate_questions_from_logs(
        self,
        logs: List[Dict[str, Any]],
        num_questions: int | None = None,
        domain_hint: str | None = "file_system",
        previous_questions: List[str] = None,
    ) -> List[Dict[str, Any]]:
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
            formatted_logs += f"Log {i+1} (Timestamp: {log['timestamp']}):\n"
            formatted_logs += f"{log['log_text']}\n\n"

        # System prompt for structured question generation
        system_prompt = """You are an expert at creating realistic and diverse Linux file system questions.
Given logs from a Linux system, generate questions that a user might ask about the SPECIFIC ACTIVITIES shown in these logs.

IMPORTANT: Each question MUST follow this EXACT format:
---
question: [The user's question here - make it natural and conversational]
type: [command OR information]
expected_response: [Detailed command with options OR comprehensive explanation]
---

Focus all questions on the path "D:\\Graduation_Project_Test_Environment" and its contents.

Guidelines for creating highly relevant and diverse questions:
1. Directly reference specific files, directories, and actions mentioned in the logs
2. Create questions that focus on the "D:\\Graduation_Project_Test_Environment" directory and its subdirectories
3. "command" type: Questions seeking specific Linux commands to accomplish tasks shown in the logs
4. "information" type: Questions seeking explanations about concepts or file system behavior evident in the logs

Your questions MUST be directly derived from the logs, such as:
- If logs show operations on files in "D:\\Graduation_Project_Test_Environment\\data", ask about those specific files
- If logs show creation of new directories, ask about making or listing directories
- If logs show file modification times, ask about checking or monitoring file changes
"""

        # Check if we have previous questions to avoid
        recent_examples = ""
        if previous_questions and len(previous_questions) > 0:
            recent_examples = (
                "RECENTLY GENERATED QUESTIONS (AVOID CREATING SIMILAR ONES):\n"
            )
            for i, q in enumerate(previous_questions[-5:]):  # Take up to 5 most recent
                recent_examples += f"{i+1}. {q}\n"
            recent_examples += "\n"

        # Human prompt with logs and specific instructions
        human_prompt = f"""Here are Linux system logs to generate questions from:

{formatted_logs}

{recent_examples}Please generate {num_questions} realistic file system questions based on THESE SPECIFIC LOGS.

IMPORTANT REQUIREMENTS:
1. All questions MUST be directly related to the activities shown in these logs
2. Reference specific files, paths, commands, and actions mentioned in the logs
3. ALWAYS focus on the "D:\\Graduation_Project_Test_Environment" directory and its contents
4. Include a mix of "command" and "information" type questions
5. For command questions, the expected_response MUST include the full command with options
6. For information questions, the expected_response MUST be a comprehensive explanation
7. Each question MUST be separated with a blank line

Examples of good questions based on sample logs:
- If logs show "created D:\\Graduation_Project_Test_Environment\\data\\temp", ask "How can I list all files in the newly created temp directory?"
- If logs show "modified D:\\Graduation_Project_Test_Environment\\config.ini", ask "How can I monitor changes to the config.ini file in real-time?"
- If logs show "deleted D:\\Graduation_Project_Test_Environment\\logs\\old_data", ask "What command would restore the deleted old_data directory if it was backed up?"
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

    def enhance_with_rag(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                4. {'Includes the exact command with proper options' if question_type == 'command' else 'Provides thorough explanation with examples'}
                
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
    ) -> List[Dict[str, Any]]:
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
    ) -> List[Dict[str, Any]]:
        """
        Generate questions specifically designed to trigger the code execution agent.
        These questions focus on file system analysis that requires running code.

        Args:
            num_questions: Number of questions to generate

        Returns:
            List of dictionaries with structured questions
        """
        system_prompt = """You are an expert at creating Linux questions that require code execution.
Generate questions about file system analysis that would require running Python or shell code to answer.

IMPORTANT: Each question MUST follow this EXACT format:
---
question: [The user's question here about file system analysis]
type: command
expected_response: [The code that would need to be executed plus explanation]
code_solution: [Python or shell code that would solve this]
---

Focus on questions about the "D:\\Graduation_Project_Test_Environment" directory and its contents.

Focus on questions that require file system analysis:
1. Finding largest/smallest files or directories in D:\\Graduation_Project_Test_Environment
2. Analyzing file types and distributions within this directory
3. Identifying duplicate files in D:\\Graduation_Project_Test_Environment\\data
4. Finding recently modified files in this directory structure
5. Analyzing disk usage patterns for D:\\Graduation_Project_Test_Environment
6. Searching for files with specific content in this directory
7. Comparing subdirectories within D:\\Graduation_Project_Test_Environment

These should be questions that would benefit from running code rather than simple Linux commands.
"""

        human_prompt = f"""Please generate {num_questions} questions about file system analysis that would require code execution.

IMPORTANT REQUIREMENTS:
1. All questions MUST follow the exact format specified
2. All questions MUST focus on the "D:\\Graduation_Project_Test_Environment" directory or its contents
3. Questions should require analysis that's easiest with Python or complex shell scripts
4. Include the code_solution field with working Python or shell code
5. Make questions specific and practical
6. Each question MUST be separated with a blank line

Examples of good questions:
- "What are the 5 largest files in D:\\Graduation_Project_Test_Environment and their sizes?"
- "How many duplicate files do I have in D:\\Graduation_Project_Test_Environment\\data?"
- "What's the distribution of file types in D:\\Graduation_Project_Test_Environment?"
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
                    
                    Code used: {tool_result.get('code', '')}
                    
                    Execution result: {tool_result.get('execution_result', '')}
                    
                    Agent analysis: {tool_result.get('agent_output', '')}
                    
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
    ) -> List[Dict[str, Any]]:
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
