import pyttsx3

print("[DEBUG] Initializing pyttsx3 engine...")
engine = pyttsx3.init()

print("[DEBUG] Engine initialized. Fetching properties...")
rate = engine.getProperty('rate')
volume = engine.getProperty('volume')
voices = engine.getProperty('voices')

print(f"[DEBUG] Rate: {rate}")
print(f"[DEBUG] Volume: {volume}")
if voices:
    print(f"[DEBUG] Voice count: {len(voices)}")
    print(f"[DEBUG] Selected voice: {voices[0].name}")
    engine.setProperty('voice', voices[0].id)

text = "Hello, this is a direct, blocking TTS test."

print(f"[DEBUG] Before say(): queuing text...")
engine.say(text)

print(f"[DEBUG] Before runAndWait(): starting playback loop...")
engine.runAndWait()

print(f"[DEBUG] After runAndWait(): playback finished.")
