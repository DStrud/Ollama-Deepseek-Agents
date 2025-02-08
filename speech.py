import random
import numpy as np
import simpleaudio as sa
import soundfile as sf

try:
    from kokoro import list_voices, KPipeline
except ImportError:
    try:
        from kokoro.voice import list_voices
        from kokoro import KPipeline
    except ImportError:
        print("Warning: 'list_voices' not found. Continuing without listing voices.")
        list_voices = lambda: []
        from kokoro import KPipeline

# -----------------------------------------------------------------------------
# Voice Lists & TTS Setup
# -----------------------------------------------------------------------------
english_voices = [
    # American English voices (female)
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    # American English voices (male)
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
    # British English voices (female)
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    # British English voices (male)
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis"
]

voices = list_voices()  # Possibly empty if listing is not supported
if voices:
    print("Available Voices:", voices)
else:
    print("No voices available (or listing not supported).")

# Global dictionary: agent_name -> assigned_voice
agent_voices = {}

def assign_voice_to_agent(agent_name):
    """Randomly picks a voice from the english_voices list for the given agent_name."""
    if agent_name not in agent_voices:
        chosen = random.choice(english_voices)
        agent_voices[agent_name] = chosen
        print(f"Assigned voice '{chosen}' to agent '{agent_name}'")

def generate_speech(text, agent_name):
    """Generate and play TTS for a given agent's text."""
    if agent_name not in agent_voices:
        print(f"No voice assigned for agent '{agent_name}'. Skipping TTS.")
        return

    voice = agent_voices[agent_name]
    try:
        speech_gen = KPipeline(lang_code="a")(text=text, voice=voice)
        speech = next(speech_gen)  # Retrieve the generated result

        # Attempt to unify results to (audio_data, sample_rate)
        if hasattr(speech, "wav") and hasattr(speech, "sr"):
            audio_data = np.array(speech.wav, dtype=np.float32)
            sample_rate = speech.sr
        elif isinstance(speech, dict) and "audio" in speech and "sample_rate" in speech:
            audio_data = np.array(speech["audio"], dtype=np.float32)
            sample_rate = speech["sample_rate"]
        elif isinstance(speech, (tuple, list)) and len(speech) >= 2:
            audio_data = np.array(speech[0], dtype=np.float32)
            sample_rate = speech[1]
        elif hasattr(speech, "output") and hasattr(speech.output, "audio"):
            audio_tensor = speech.output.audio
            audio_data = audio_tensor.detach().cpu().numpy() if hasattr(audio_tensor, "detach") else np.array(audio_tensor, dtype=np.float32)
            sample_rate = 22050  # default sample rate
        else:
            raise ValueError("Unrecognized result format from Kokoro pipeline.")

        # Save to WAV, then play
        output_filename = "output.wav"
        sf.write(output_filename, audio_data, sample_rate)
        wave_obj = sa.WaveObject.from_wave_file(output_filename)
        play_obj = wave_obj.play()
        play_obj.wait_done()

    except Exception as e:
        print(f"Error generating speech for '{agent_name}': {e}")
