import random
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from tracer.config import LogDomain

from os_assistant.config.settings import MODEL_BASE_URL
from os_assistant.tools.agentic_rag.application.search import search_logs
from os_assistant.tools.code_agent.wrapper import code_execute_tool

from ..config import DATASET_LLM_MODEL, MAX_QUESTIONS_PER_LOG, MIN_QUESTIONS_PER_LOG


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
    ) -> List[Dict[str, Any]]:
        """
        Generate structured questions based on log content with explicit type and expected response.

        Args:
            logs: List of log dictionaries
            num_questions: Optional number of questions to generate (random if None)
            domain_hint: Optional domain hint to focus question generation

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
Given logs from a Linux system, generate questions that a user might ask.

IMPORTANT: Each question MUST follow this EXACT format:
---
question: [The user's question here - make it natural and conversational]
type: [command OR information]
expected_response: [Detailed command with options OR comprehensive explanation]
---

Guidelines for creating diverse questions:
1. "command" type: Questions seeking specific Linux commands to accomplish tasks
2. "information" type: Questions seeking explanations about concepts or file system behavior

Include a variety of question formulations:
- "I can't access..." (problems users are having)
- "How do I..." (seeking instructions)
- "Can you help me..." (requesting assistance)
- "What's the best way to..." (seeking recommendations)
- "Is it possible to..." (checking feasibility)

Make questions realistic, practical, and focused on file system operations:
- File permissions and ownership
- Finding, creating, copying, moving, and deleting files
- Directory navigation and structure
- File content searching and manipulation
- Disk usage and storage
"""

        # Human prompt with logs and specific instructions
        human_prompt = f"""Here are Linux system logs to generate questions from:

{formatted_logs}

Please generate {num_questions} realistic file system questions based on these logs.

IMPORTANT REQUIREMENTS:
1. All questions MUST follow the exact format specified
2. All questions MUST focus on file system operations
3. Include a mix of "command" and "information" type questions
4. For command questions, the expected_response MUST include the full command with options
5. For information questions, the expected_response MUST be a comprehensive explanation
6. Reference specific details from the logs when possible
7. Each question MUST be separated with a blank line

Remember to create questions that users would naturally ask, like:
- "I can't access my files in /home/user, can you help?"
- "How do I find all .txt files modified in the last week?"
- "What's the meaning of 'drwxr-xr-x' in the output?"
"""

        # Generate questions
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
                    "source_logs": [log["log_number"] for log in logs],
                    "domain": domain_hint or logs[0].get("domain", "file_system"),
                    "timestamps": [log["timestamp"] for log in logs],
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
1. Common file operations (find, copy, move, delete)
2. Permission issues and ownership
3. Directory structure and navigation
4. File content searching and manipulation
5. Disk space and storage management

Include a variety of question formulations:
- "I can't access..." (problems users are having)
- "How do I..." (seeking instructions)
- "Can you help me..." (requesting assistance)
- "What's the best way to..." (seeking recommendations)
- "Is it possible to..." (checking feasibility)
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
- "How do I find all files larger than 100MB in my home directory?"
- "I can't figure out why I'm getting 'permission denied' when trying to edit /etc/hosts"
- "What's the difference between hard links and symbolic links?"
- "How can I see which directories are taking up the most space?"
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
                    "generated_type": "random",
                }
            )

        return result

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

Focus on questions that require file system analysis:
1. Finding largest/smallest files or directories
2. Analyzing file types and distributions
3. Identifying duplicate files
4. Finding recently modified files
5. Analyzing disk usage patterns
6. Searching for files with specific content
7. Comparing directories

These should be questions that would benefit from running code rather than simple Linux commands.
"""

        human_prompt = f"""Please generate {num_questions} questions about file system analysis that would require code execution.

IMPORTANT REQUIREMENTS:
1. All questions MUST follow the exact format specified
2. Questions should require analysis that's easiest with Python or complex shell scripts
3. Include the code_solution field with working Python or shell code
4. Make questions specific and practical
5. Each question MUST be separated with a blank line

Examples of good questions:
- "What are the 5 largest files in my home directory and their sizes?"
- "How many duplicate files do I have in my Downloads folder?"
- "What's the distribution of file types in my Documents directory?"
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
                    "generated_type": "code_execution",
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
