# app.py

from flask import Flask, render_template
from flask_socketio import SocketIO
import time

from agents import (
    PlannerAgent,
    GenericAgent,
    agent_memory
)
from memory import save_memory
from speech import generate_speech, assign_voice_to_agent

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Our global registry of agent_name -> Agent instance
active_agents = {}

# Global message queue
message_queue = []

def get_messages(recipient):
    """Retrieve messages for a given recipient from the global queue."""
    msgs = [msg for msg in message_queue if msg["to"] == recipient]
    for m in msgs:
        message_queue.remove(m)
    return msgs

def send_message(sender, recipient, content):
    """Send a message from one agent to another."""
    message = {"from": sender, "to": recipient, "content": content}
    message_queue.append(message)
    # Broadcast to the front-end
    socketio.emit("new_message", message)
    # TTS for the sender's text (optional)
    generate_speech(content, sender)
    # Make sure the agents exist
    if sender not in active_agents:
        # Possibly create an agent if needed
        pass
    if recipient not in active_agents:
        pass

# --- Agent Spawner Callback ---
spawned_agent_count = 0

def spawn_agent_with_role(role):
    """
    Creates a new GenericAgent with the given role,
    returns its newly assigned name.
    """
    global spawned_agent_count
    spawned_agent_count += 1
    agent_name = f"{role}_{spawned_agent_count}"  # e.g. "Researcher_1"

    # Create the agent
    new_agent = GenericAgent(agent_name, role)
    active_agents[agent_name] = new_agent

    # Assign a voice
    assign_voice_to_agent(agent_name)
    print(f"[Planner] Spawned a new agent: {agent_name} (role: {role})")
    return agent_name

# Create the Planner once at startup
def create_planner():
    planner_name = "Planner"
    planner = PlannerAgent(
        name=planner_name, 
        role="Planner", 
        agent_spawner_callback=spawn_agent_with_role
    )
    active_agents[planner_name] = planner
    assign_voice_to_agent(planner_name)
    return planner_name

planner_name = create_planner()

def run_agents(goal):
    """Kick off the conversation by sending the user's goal to the Planner."""
    send_message("User", planner_name, goal)

    max_iterations = 20
    iterations = 0

    while message_queue and iterations < max_iterations:
        # Let each agent process incoming messages
        for agent in list(active_agents.values()):
            agent.communicate(get_messages, send_message)
        iterations += 1
        time.sleep(0.3)

    # Save memory at the end
    save_memory(agent_memory)
    print("Conversation ended or max iterations reached.")

    # Debug: print final memories
    for aname, agent in active_agents.items():
        print(f"{aname} memory:\n{agent.get_memory()}")
        print("-" * 50)

@app.route("/")
def home():
    return render_template("index.html")

@socketio.on("start_agents")
def handle_start_agents(data):
    goal = data.get("goal", "")
    if goal:
        run_agents(goal)
    else:
        socketio.emit("error", {"error": "No goal provided."})

if __name__ == "__main__":
    socketio.run(app, debug=True)
