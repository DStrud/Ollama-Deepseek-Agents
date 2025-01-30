import requests
import json
import os
import time
from flask import Flask, render_template
from flask_socketio import SocketIO

# Flask Setup
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable WebSocket support

# Ollama API
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:14b"
MEMORY_FILE = "memory.json"

# Load memory from JSON file
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as file:
            return json.load(file)
    return {}

# Save memory to JSON file
def save_memory(memory_data):
    with open(MEMORY_FILE, "w") as file:
        json.dump(memory_data, file, indent=4)

# Initialize memory and messaging
agent_memory = load_memory()
message_queue = []

def send_message(sender, recipient, content):
    """Sends a message between agents and updates the UI via WebSocket."""
    message = {"from": sender, "to": recipient, "content": content}
    message_queue.append(message)
    socketio.emit("new_message", message)  # Send live update to frontend

def get_messages(recipient):
    """Retrieves messages for a specific agent."""
    messages = [msg for msg in message_queue if msg["to"] == recipient]
    for msg in messages:
        message_queue.remove(msg)
    return messages

# Query Ollama API
def query_ollama(prompt):
    data = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result.get("response", "No response from Ollama.")
    else:
        return f"Error: {response.status_code} - {response.text}"

# Base Agent class
class Agent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.memory = agent_memory.get(self.name, [])
        agent_memory[self.name] = self.memory

    def remember(self, message):
        """Stores message in memory and saves it."""
        self.memory.append(message)
        if len(self.memory) > 5:
            self.memory = self.memory[-5:]
        agent_memory[self.name] = self.memory
        save_memory(agent_memory)

    def get_memory(self):
        """Retrieves conversation history."""
        return "\n".join(self.memory)

    def communicate(self):
        """Processes messages and responds."""
        messages = get_messages(self.name)
        if not messages:
            return None

        responses = []
        for message in messages:
            response = self.respond(message["content"])
            responses.append(response)
            send_message(self.name, message["from"], response)

        return responses

    def respond(self, message):
        """Processes a message (to be implemented by subclasses)."""
        raise NotImplementedError("Each agent must implement its own method.")

# Define agents
class PlannerAgent(Agent):
    def communicate(self, goal):
        """The Planner breaks the task into steps and keeps agents on track."""
        send_message("Planner", "Researcher", f"Please research: {goal}")

    def review_responses(self):
        """Check if agent responses match the original task."""
        messages = get_messages(self.name)  # Get messages sent to the Planner

        for message in messages:
            sender = message["from"]
            response = message["content"]

            # Check if response actually answers the request
            if self.is_off_topic(response):
                send_message(self.name, sender, "Your response went off-topic. Please refocus on the main task.")
            else:
                # If it's correct, send it to the next step
                if sender == "Researcher":
                    send_message(self.name, "Writer", f"Use this research: {response}")
                elif sender == "Writer":
                    send_message(self.name, "Reviewer", f"Review this document: {response}")

    def is_off_topic(self, response):
        """Simple check to see if the response is too broad or off-topic."""
        # If response is way too long or contains speculative text, flag it
        if len(response.split()) > 250 or "maybe" in response or "what if" in response:
            return True
        return False

class ResearcherAgent(Agent):
    def respond(self, task):
        time.sleep(1)  # Simulate processing time
        memory = self.get_memory()
        prompt = f"Your memory:\n{memory}\nResearch and summarize:\n{task}"
        response = query_ollama(prompt)
        self.remember(f"Research: {response}")
        send_message(self.name, "Writer", f"Here's the research: {response}")
        return response

class WriterAgent(Agent):
    def respond(self, research_data):
        time.sleep(1)  # Simulate processing time
        memory = self.get_memory()
        prompt = f"Your memory:\n{memory}\nWrite a structured document:\n{research_data}"
        response = query_ollama(prompt)
        self.remember(f"Draft: {response}")
        send_message(self.name, "Reviewer", f"Please review this document: {response}")
        return response

class ReviewerAgent(Agent):
    def respond(self, draft):
        time.sleep(1)  # Simulate processing time
        memory = self.get_memory()
        prompt = f"Your memory:\n{memory}\nReview this document for errors:\n{draft}"
        response = query_ollama(prompt)
        self.remember(f"Review: {response}")
        return response

# Create agents
planner = PlannerAgent("Planner", "Breaks down tasks")
researcher = ResearcherAgent("Researcher", "Gathers data")
writer = WriterAgent("Writer", "Creates a document")
reviewer = ReviewerAgent("Reviewer", "Checks for errors")

# Process messages
def run_agents(goal):
    """Main loop to process agent messages until all tasks are completed."""
    send_message("User", "Planner", goal)  # Start by asking the planner
    
    while message_queue:
        planner.communicate(goal)  # Pass the goal so it doesn't break
        planner.review_responses()  # Check and correct responses if needed
        researcher.communicate()
        writer.communicate()
        reviewer.communicate()

@app.route("/")
def home():
    return render_template("index.html")

@socketio.on("start_agents")
def handle_start_agents(data):
    goal = data["goal"]
    run_agents(goal)

if __name__ == "__main__":
    socketio.run(app, debug=True)
