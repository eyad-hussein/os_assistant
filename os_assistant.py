import json
import logging
import asyncio
import random
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Literal, Union

from pydantic import BaseModel, Field, ValidationError
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai.flow import Flow, listen, start
from crewai.flow.flow import Flow, start, listen, router, or_

# Import the RAG manager
from rag_manager import get_rag_manager

# Initialize LLM
llm = LLM(model="ollama/llama3", base_url="https://6346-34-75-218-191.ngrok-free.app")

# Get RAG manager instance
rag_manager = get_rag_manager()

# Define available domains
domains_above = ["filesystem", "users", "packages", "network"]

# Create domain logs directory
logs_dir = "domain_logs"
os.makedirs(logs_dir, exist_ok=True)

# Function to generate random timestamps within a recent time period
def random_timestamp():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    time_delta = end_date - start_date
    random_seconds = random.randint(0, int(time_delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)

# Generate domain-specific logs with timestamps
filesystem_logs = [
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Created new directory /home/user/projects"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Moved ahmed.pdf: mv /home/user/ahmed.pdf /home/user/Documents/"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Changed permissions on /etc/config to read-only"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Cleaned up temporary files in /tmp directory"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Created system backup"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "My cat pictures are stored in /home/user/Pictures/cats folder"
    }
]

users_logs = [
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Changed password for user sarah"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Added new user mohammed"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "my password is 213214"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Created guest account for visitor"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "User mohammed logged in from 192.168.1.5"
    }
]

packages_logs = [
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Updated all packages with apt update && apt upgrade"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Installed nginx"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Removed obsolete packages with apt autoremove"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Installed ollama for local LLM support"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Configured python-venv"
    }
]

network_logs = [
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Created new network 'Home_Network'"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Configured static IP address 192.168.1.100"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Set up port forwarding on router for SSH access"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "DNS server switched to 8.8.8.8 and 8.8.4.4"
    },
    {
        "timestamp": random_timestamp().isoformat(),
        "log": "Pinged google.com to check internet connectivity"
    }
]

# Save logs to domain-specific JSON files
domains_data = {
    "filesystem": sorted(filesystem_logs, key=lambda x: x["timestamp"], reverse=True),
    "users": sorted(users_logs, key=lambda x: x["timestamp"], reverse=True),
    "packages": sorted(packages_logs, key=lambda x: x["timestamp"], reverse=True),
    "network": sorted(network_logs, key=lambda x: x["timestamp"], reverse=True)
}

# Write each domain's logs to its own JSON file
for domain, logs in domains_data.items():
    file_path = os.path.join(logs_dir, f"{domain}_logs.json")
    with open(file_path, 'w') as json_file:
        json.dump(logs, json_file, indent=4)

# Display sample of logs from each domain
for domain, logs in domains_data.items():
    print(f"{domain.upper()} - Latest logs:")
    for log in logs[:2]:  # Display only top 2 logs
        time_str = datetime.fromisoformat(log["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{time_str}: {log['log']}")
    print()

# --- Tool Definitions ---

@tool
def retrieve_domain_information(domain: str, query: str, max_results: int = 3) -> str:
    """Retrieves relevant information from a domain using semantic search.
    
    Args:
        domain: Domain to search (filesystem, users, packages, network)
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        Formatted string with search results
    """
    # Handle case when max_results is passed as a dictionary instead of int
    if isinstance(max_results, dict) and 'type' in max_results and max_results['type'] == 'int':
        max_results = 3  # Default to 3 if we get a schema dict instead of value
    
    # Ensure max_results is an integer
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 3  # Default to 3 if conversion fails
        
    return rag_manager.retrieve_formatted(domain, query, max_results)

@tool
def search_across_domains(query: str, domains: List[str], max_per_domain: int = 2) -> str:
    """Search for relevant information across multiple domains.
    
    Args:
        query: Search query
        domains: List of domains to search
        max_per_domain: Maximum results per domain
        
    Returns:
        Formatted string with search results
    """
    # Similarly handle potential dict parameters
    if isinstance(max_per_domain, dict) and 'type' in max_per_domain and max_per_domain['type'] == 'int':
        max_per_domain = 2  # Default if we get a schema dict
        
    try:
        max_per_domain = int(max_per_domain)
    except (TypeError, ValueError):
        max_per_domain = 2  # Default if conversion fails
        
    domain_list = [d.strip() for d in domains]
    return rag_manager.multi_domain_retrieve_formatted(query, domain_list, max_per_domain)

# --- Model Definitions ---

class DomainAnalysis(BaseModel):
    """Model for domain analysis results"""
    domains: List[str] = Field(..., description="List of relevant domains")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    reasoning: str = Field(..., description="Explanation of why these domains were selected")

class ContextResult(BaseModel):
    """Model for context retrieval results"""
    context: str = Field(..., description="Retrieved context")
    domain: str = Field(..., description="Domain of the context")

class QueryTypeResult(BaseModel):
    """Model for query type classification"""
    query_type: Literal["command", "information"] = Field(..., 
                description="Type of query: command or information")
    reasoning: str = Field(..., description="Explanation for the classification")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in classification")

class CommandResponse(BaseModel):
    """Model for command generation response"""
    command: str = Field(..., description="Linux command to execute")
    explanation: str = Field(..., description="Explanation of the command")
    security_notes: str = Field(default="", description="Security considerations")

class InformationResponse(BaseModel):
    """Model for information retrieval response"""
    answer: str = Field(..., description="Answer to the query")
    sources: List[str] = Field(default_factory=list, description="Sources of information")

class FinalResult(BaseModel):
    """Final result combining all outputs"""
    query: str = Field(..., description="Original query")
    domains: List[str] = Field(..., description="Domains analyzed")
    response_type: Literal["command", "information"] = Field(..., 
                    description="Type of response provided")
    response: Union[CommandResponse, InformationResponse] = Field(..., 
                    description="The response content")
    context_summary: str = Field(..., description="Summary of context used")

# --- Agent Definitions ---

domain_agent = Agent(
    role="Domain Classifier",
    goal="Accurately categorize Linux prompts into technical domains",
    backstory="""I am a specialized Linux domain expert with deep knowledge of 
    system architecture. My expertise allows me to quickly identify which technical 
    domain a query belongs to, even when it spans multiple domains. I understand 
    the relationships between filesystem operations, user management, package handling, 
    and network configuration.""",
    verbose=True,
    llm=llm
)

domain_task = Task(
    description="""Analyze this Linux query: '{prompt}' and identify the most relevant domains from: {domains}.
    
    Guidelines:
    1. Consider that queries may relate to multiple domains
    2. Focus on the primary intent of the query
    3. Include only truly relevant domains (don't force matches)
    4. Provide your reasoning for the selected domains
    
    Domains overview:
    - filesystem: File operations, directories, permissions, storage
    - users: User accounts, passwords, authentication, user groups
    - packages: Software installation, updates, package management
    - network: Connectivity, IP configuration, networking tools
    """,
    expected_output=DomainAnalysis.schema_json(),
    agent=domain_agent,
    output_json=DomainAnalysis
)

context_agent = Agent(
    role="Technical Context Retriever",
    goal="Fetch relevant technical documentation and examples with timestamps",
    backstory="""Expert in searching and retrieving precise technical information.
    I specialize in finding the most relevant and recent logs across different system domains.""",
    tools=[retrieve_domain_information],
    verbose=True,
    llm=llm
)

context_task = Task(
    description="Retrieve context for domain: {domain} related to: {prompt}. Find the most relevant information.",
    expected_output=ContextResult.schema_json(),
    agent=context_agent,
    output_json=ContextResult
)

query_classifier_agent = Agent(
    role="Query Type Classifier",
    goal="Determine if a query requires a command or information",
    backstory="""I analyze user queries to determine if they're seeking a command to execute 
    or information to understand. I look for intent markers and contextual clues to classify queries accurately.""",
    tools=[],
    verbose=True,
    llm=llm
)

query_classifier_task = Task(
    description="""Classify this query: '{prompt}' as either requiring a 'command' or 'information'.
    
    Context from domains:
    {context}
    
    Guidelines:
    - 'command': User wants to perform an action or needs a Linux command to execute
    - 'information': User wants facts, explanations, or understanding about something
    
    Provide your reasoning for the classification.
    """,
    expected_output=QueryTypeResult.schema_json(),
    agent=query_classifier_agent,
    output_json=QueryTypeResult
)

command_agent = Agent(
    role="Linux Command Architect",
    goal="Generate safe and effective Linux commands specific to the user's system",
    backstory="Experienced system administrator specializing in secure command generation tailored to each user's specific environment and needs.",
    tools=[retrieve_domain_information],
    verbose=True,
    llm=llm
)

command_task = Task(
    description="""Generate a Linux command for: '{prompt}' based on these domains: {domains} and the following context:
    {context}
    
    IMPORTANT: The context contains information from the user's actual system.
    Your command should be specifically tailored to the user's environment based on the context.
    
    Your response must include:
    1. A single, executable Linux command that addresses the user's request
    2. A brief explanation of what the command does ON THE USER'S SPECIFIC SYSTEM
    3. Any security considerations related to their environment
    
    Use specific paths, usernames, and system details from the context in your command when applicable.
    Refer to the user's environment directly (e.g., "your system", "your files", etc.).
    """,
    expected_output=CommandResponse.schema_json(),
    agent=command_agent,
    output_json=CommandResponse
)

information_agent = Agent(
    role="Information Specialist",
    goal="Provide personalized information specific to the user's system",
    backstory="""I specialize in retrieving and presenting accurate information about the user's specific system.
    I understand that logs and data I access belong to the user's actual device, and I provide answers that are
    specific to their environment, not generic knowledge.""",
    tools=[retrieve_domain_information, search_across_domains],
    verbose=True,
    llm=llm
)

information_task = Task(
    description="""Answer this question: '{prompt}' using the following context:
    {context}
    
    IMPORTANT INSTRUCTIONS:
    - The context contains information from the user's ACTUAL SYSTEM, not general knowledge
    - Treat this as personalized information about THEIR specific device/environment
    - Reference specific details from the context (timestamps, paths, usernames, etc.)
    - Use phrases like "your system", "on your device", "in your configuration", etc.
    
    YOU MUST RESPOND WITH JSON IN THIS EXACT FORMAT:
    {{
        "answer": "Your answer that clearly references the user's specific system information",
        "sources": ["Source 1", "Source 2"]
    }}
    
    DO NOT include any text before or after the JSON.
    DO NOT include explanations about the JSON format.
    ONLY OUTPUT VALID JSON that matches the specified format.
    """,
    expected_output=InformationResponse.schema_json(),
    agent=information_agent,
    output_json=InformationResponse
)

# --- Flow State ---

class LinuxAssistantState(BaseModel):
    """State for the Linux assistant flow"""
    # Input
    prompt: str = ""
    domains: List[str] = Field(default_factory=list)
    
    # Domain analysis
    domain_analysis: Optional[DomainAnalysis] = None
    
    # Context retrieval
    contexts: Dict[str, str] = Field(default_factory=dict)
    current_domain_index: int = 0
    
    # Query classification
    query_type: Optional[QueryTypeResult] = None
    
    # Response generation
    command_response: Optional[CommandResponse] = None
    information_response: Optional[InformationResponse] = None
    
    # Final result
    final_result: Optional[FinalResult] = None

# --- Flow Implementation ---

class LinuxAssistantFlow(Flow[LinuxAssistantState]):
    """Flow for Linux assistant with router-based control flow"""

    @start()
    def get_user_input(self):
        """Get input from user to start the flow"""
        self.state.prompt = input("Enter your Linux question: ")
        if not self.state.prompt:
            self.state.prompt = "where are my cat pictures stored?"
        print(f"\nProcessing query: '{self.state.prompt}'")
        self.state.domains = domains_above
        return "domain_analysis"

    @listen(get_user_input)
    def domain_analysis(self, state: LinuxAssistantState):
        """Analyze domains relevant to the query"""
        print("\nAnalyzing query domains...")
        domain_response = domain_agent.execute_task(
            domain_task,
            context=json.dumps({
                "prompt": self.state.prompt,
                "domains": ", ".join(self.state.domains)
            })
        )
        
        self.state.domain_analysis = DomainAnalysis.parse_raw(domain_response)
        print(f"Domains identified: {self.state.domain_analysis.domains}")
        print(f"Confidence: {self.state.domain_analysis.confidence}")
        print(f"Reasoning: {self.state.domain_analysis.reasoning}")
        
        # Initialize context retrieval
        self.state.current_domain_index = 0
        
        return "router"

    @router(or_(domain_analysis, "context_retrieval", "query_classification", "generate_command","generate_information"))
    def router(self, state: LinuxAssistantState):
        """Route the flow based on current state"""
        # If we have domains to process, continue with context retrieval
        if self.state.domain_analysis and self.state.current_domain_index < len(self.state.domain_analysis.domains):
            return "context_retrieval_router"
            
        # If all domains processed but query type not determined, go to classification
        elif self.state.domain_analysis and not self.state.query_type:
            return "query_classification_from_router"
            
        # If query type determined but no response generated yet
        elif self.state.query_type and self.state.query_type.query_type == "command" and not self.state.command_response:
            return "generate_command_router"
            
        elif self.state.query_type and self.state.query_type.query_type == "information" and not self.state.information_response:
            return "generate_information_router"
            
        # If all steps completed, finish
        else:
            return "prepare_final_result_from_router"

    @listen("context_retrieval_router")
    def context_retrieval(self, state: LinuxAssistantState):
        """Retrieve context for the current domain"""
        current_domain = self.state.domain_analysis.domains[self.state.current_domain_index]
        print(f"\nRetrieving context for domain: {current_domain}")
        
        # Explicitly set max_results as integer in context
        context_response = context_agent.execute_task(
            context_task,
            context=json.dumps({
                "domain": current_domain,
                "prompt": self.state.prompt,
                "max_results": 3  # Explicitly provide as integer
            })
        )
        
        result = ContextResult.parse_raw(context_response)
        self.state.contexts[current_domain] = result.context
        
        print(f"Retrieved context from {current_domain}")
        
        # Move to next domain
        self.state.current_domain_index += 1
        
        return "router"

    @listen("query_classification_from_router")
    def query_classification(self, state: LinuxAssistantState):
        """Classify the query type (command or information)"""
        print("\nClassifying query type...")
        
        # Prepare combined context
        combined_context = ""
        for domain, context in self.state.contexts.items():
            combined_context += f"--- {domain.upper()} DOMAIN ---\n{context}\n\n"
        
        query_response = query_classifier_agent.execute_task(
            query_classifier_task,
            context=json.dumps({
                "prompt": self.state.prompt,
                "context": combined_context
            })
        )
        
        self.state.query_type = QueryTypeResult.parse_raw(query_response)
        
        print(f"Query classified as: {self.state.query_type.query_type}")
        print(f"Reasoning: {self.state.query_type.reasoning}")
        
        return "router"

    @listen("generate_command_router")
    def generate_command(self, state: LinuxAssistantState):
        """Generate a command response"""
        print("\nGenerating Linux command...")
        
        # Prepare combined context
        combined_context = ""
        for domain, context in self.state.contexts.items():
            combined_context += f"--- {domain.upper()} DOMAIN ---\n{context}\n\n"
        
        command_response = command_agent.execute_task(
            command_task,
            context=json.dumps({
                "prompt": self.state.prompt,
                "domains": ", ".join(self.state.domain_analysis.domains),
                "context": combined_context
            })
        )
        
        self.state.command_response = CommandResponse.parse_raw(command_response)
        
        # Ensure the explanation is personalized if not already
        if not any(phrase in self.state.command_response.explanation.lower() for phrase in 
                  ["your", "you", "on your", "in your"]):
            self.state.command_response.explanation = f"On your specific system, {self.state.command_response.explanation[0].lower()}{self.state.command_response.explanation[1:]}"
        
        print(f"Generated command: {self.state.command_response.command}")
        
        return "router"

    @listen("generate_information_router")
    def generate_information(self, state: LinuxAssistantState):
        """Generate an information response"""
        print("\nGenerating information response...")
        
        # Prepare combined context
        combined_context = ""
        for domain, context in self.state.contexts.items():
            combined_context += f"--- {domain.upper()} DOMAIN ---\n{context}\n\n"
        
        try:
            # First attempt: Standard execution
            info_response = information_agent.execute_task(
                information_task,
                context=json.dumps({
                    "prompt": self.state.prompt,
                    "context": combined_context
                })
            )
            
            # Try to parse the response as JSON
            try:
                self.state.information_response = InformationResponse.parse_raw(info_response)
                
                # Ensure the answer is personalized if not already
                if not any(phrase in self.state.information_response.answer.lower() for phrase in 
                         ["your", "you", "on your", "in your"]):
                    self.state.information_response.answer = f"On your system, {self.state.information_response.answer[0].lower()}{self.state.information_response.answer[1:]}"
                
                print(f"Successfully generated JSON response")
                return "router"
            except Exception as e:
                print(f"Warning: Failed to parse information response as JSON: {str(e)}")
            
            # Second attempt processing code remains the same
            # ...existing code...
            
        except Exception as ex:
            print(f"Error in information generation: {str(ex)}")
            self.state.information_response = InformationResponse(
                answer="I encountered an error while processing information about your specific system. Please try rephrasing your query.",
                sources=["Error recovery"]
            )
        
        return "router"

    @listen('display_result')
    def display_result(self, state: LinuxAssistantState):
        """Display the final result to the user"""
        if not self.state.final_result:
            print("\nError: No final result generated.")
            return "end"
        
        print("\n" + "="*60)
        print("LINUX ASSISTANT RESULT")
        print("="*60)
        
        print(f"Query: {self.state.final_result.query}")
        print(f"Domains analyzed: {', '.join(self.state.final_result.domains)}")
        
        if self.state.final_result.response_type == "command":
            command = self.state.final_result.response
            print("\nCOMMAND FOR YOUR SYSTEM:")
            print(f"$ {command.command}")
            print("\nEXPLANATION:")
            print(command.explanation)
            if command.security_notes:
                print("\nSECURITY NOTES:")
                print(command.security_notes)
        else:
            info = self.state.final_result.response
            print("\nABOUT YOUR SYSTEM:")
            print(info.answer)
            if info.sources:
                print("\nSOURCES FROM YOUR SYSTEM:")
                for source in info.sources:
                    print(f"- {source}")
        
        print("\n" + "="*60)
        return "end"

    @listen("end")
    def end(self, state: LinuxAssistantState):
        """End the flow and return the state"""
        print("Assistant process complete.")
        return self.state


async def run_linux_assistant():
    """Run the Linux assistant flow"""
    flow = LinuxAssistantFlow()
    await flow.kickoff_async()


if __name__ == "__main__":
    # Generate a visualization of the flow
    LinuxAssistantFlow().plot("linux_assistant_flow")
    
    # Run the assistant
    asyncio.run(run_linux_assistant())
