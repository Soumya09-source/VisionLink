from collections import defaultdict

# Lower number = higher priority. Unknown objects default to 99.
PRIORITIES = {
    "person": 1,
    "car": 2, "bus": 2, "truck": 2, "bicycle": 2,
    "chair": 3, "table": 3
}

def filter_and_group_detections(detections, last_spoken, now, cooldown):
    """
    Groups detections by class name and direction.
    Ignores classes that were spoken too recently.
    Returns: dict {class_name: set_of_directions}
    """
    groups = defaultdict(set)
    for d in detections:
        cls_name = d["class_name"]
        
        # 1a. Avoid repetition using the cooldown dictionary
        if now - last_spoken.get(cls_name, 0.0) < cooldown:
            continue
            
        # 1b. Group multiple instances of the same object by direction
        groups[cls_name].add(d["direction"])
        
    return groups

def format_direction_string(directions):
    """
    Converts a set of directions into a natural phrase.
    e.g., {"left", "center"} -> "on the left and center"
    e.g., {"center"} -> "in front"
    """
    dirs = list(directions)
    if len(dirs) == 1:
        return "in front" if dirs[0] == "center" else f"on the {dirs[0]}"
    
    return "on the " + " and ".join(dirs)

def generate_natural_sentence(grouped_detections):
    """
    Generates a natural sentence prioritizing important objects.
    Returns: (sentence_string, list_of_spoken_classes)
    """
    if not grouped_detections:
        return None, []
        
    # Sort classes so important objects (like 'person') are spoken first
    sorted_classes = sorted(list(grouped_detections.keys()), 
                            key=lambda x: PRIORITIES.get(x, 99))
    
    # Cap at 2 object types per sentence to avoid overwhelming the user
    top_classes = sorted_classes[:2]
    
    phrases = []
    for cls_name in top_classes:
        dir_str = format_direction_string(grouped_detections[cls_name])
        
        # Simple grammar fix: pluralize if seen in multiple directions
        is_plural = len(grouped_detections[cls_name]) > 1
        noun = f"{cls_name}s" if is_plural else cls_name
        
        phrases.append(f"{noun} {dir_str}")
        
    sentence = ", ".join(phrases)
    # Capitalize the first letter for neatness
    sentence = sentence[0].upper() + sentence[1:]
    
    return sentence, top_classes

def process_and_speak(detections, tts, last_spoken, now, cooldown=5.0):
    """
    Full pipeline: Filters -> Groups -> Generates Sentence -> Speaks -> Updates Cooldown.
    """
    grouped = filter_and_group_detections(detections, last_spoken, now, cooldown)
    
    sentence, spoken_classes = generate_natural_sentence(grouped)
    
    if sentence:
        tts.speak(sentence)
        print(f"[TTS Out] {sentence}")  # Useful for debugging
        
        # Update cooldown *only* for the classes we just spoke
        for cls_name in spoken_classes:
            last_spoken[cls_name] = now
