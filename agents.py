import time
import re
import requests
from memory import load_memory, save_memory

# -----------------------------------------------------------------------------
# OLLAMA/LLM Configuration
# -----------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:14b"

def query_ollama(prompt):
    """Sends a prompt to the local OLLAMA server and returns the text response."""
    data = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response from Ollama.")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Exception during API request: {e}"

def clean_response(response_text):
    """Removes any hidden <think>...</think> sections from the response."""
    return re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

# -----------------------------------------------------------------------------
# Agent Memory
# -----------------------------------------------------------------------------
agent_memory = load_memory()

# -----------------------------------------------------------------------------
# Base Agent Class
# -----------------------------------------------------------------------------
class Agent:
    def __init__(self, name, role):
        self.name = name
        self.role = role  # e.g. "Planner", "Researcher", "Writer", or custom
        self.memory = agent_memory.get(self.name, [])
        agent_memory[self.name] = self.memory

    def remember(self, message):
        """Save the last few interactions in the memory for context."""
        self.memory.append(message)
        if len(self.memory) > 5:
            self.memory = self.memory[-5:]
        agent_memory[self.name] = self.memory
        save_memory(agent_memory)

    def get_memory(self):
        """Return a string representation of this agent's memory."""
        return "\n".join(self.memory)

    def respond(self, message, sender):
        """
        Default fallback. Specialized agents override this.
        Return (None, None) if no response is needed.
        """
        return (None, None)

    def communicate(self, get_messages, send_message):
        """
        Called each loop iteration. Retrieves messages for this agent 
        and calls respond() for each.
        """
        messages = get_messages(self.name)
        for msg in messages:
            target, response = self.respond(msg["content"], msg["from"])
            if target and response:
                send_message(self.name, target, response)

# -----------------------------------------------------------------------------
# GenericAgent: Its role is assigned at creation time, but the logic is always:
#  "Use the role to shape how you process the prompt via the LLM."
# -----------------------------------------------------------------------------
class GenericAgent(Agent):
    def respond(self, message, sender):
        # The role field might be "Researcher", "Writer", "Proofreader", etc.
        # We'll embed it into a system-style prompt.
        system_prompt = (
            f"You are a helpful AI with the role: {self.role}.\n"
            "Use your expertise in this role to respond appropriately."
        )
        full_prompt = f"{system_prompt}\n\nUser says: {message}\nYour response:"
        llm_output = query_ollama(full_prompt)
        llm_output = clean_response(llm_output)

        self.remember(f"Received: {message}")
        self.remember(f"Response: {llm_output}")

        # In many setups, you'd parse the LLM output to see if it references
        # some next step. For simplicity, we'll just return it back to the sender.
        return (sender, llm_output)

# -----------------------------------------------------------------------------
# PlannerAgent: the only agent with a truly custom `respond` method that can
# spawn new agents with custom roles.
# -----------------------------------------------------------------------------
class PlannerAgent(Agent):
    def __init__(self, name, role, agent_spawner_callback):
        """
        agent_spawner_callback is a function that takes a role string
        and returns a new agent name. (It also creates and stores the agent
        in the global dictionary.)
        """
        super().__init__(name, role)
        self.agent_spawner_callback = agent_spawner_callback
        self.subtasks_created = False

    def respond(self, message, sender):
        """
        - On first user message: parse the goal.
        - Decide to create or reuse sub-agents with specific roles (Researcher, etc.).
        - Send them tasks.
        """
        if sender == "User" and not self.subtasks_created:
            self.remember(f"User goal: {message}")

            # Example: we create a "Researcher" role to gather info.
            researcher_agent_name = self.agent_spawner_callback("Researcher")
            # Then we create a "Writer" role to produce content from that research.
            writer_agent_name = self.agent_spawner_callback("Writer")

            self.subtasks_created = True

            # Next step: instruct the Researcher
            # We can just pass the user's message as the "research topic"
            return (researcher_agent_name, f"Please research: {message}")

        # We might add more logic if the Planner needs to handle second-phase tasks,
        # e.g., collecting results from the Researcher, forwarding to the Writer, etc.
        # For now, we do nothing else if we already created subtasks.
        return (None, None)

