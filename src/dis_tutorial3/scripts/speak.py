#!/usr/bin/env python3

import sys
import argparse
import pygame
from gtts import gTTS
from io import BytesIO
import time

def speak(text, lang='en', slow=False):
    """
    Speak the provided text using Google Text-to-Speech (gTTS) and pygame 
    without creating a temporary file on disk.
    
    Args:
        text (str): The text to be spoken
        lang (str): Language code (default: 'en')
        slow (bool): Whether to speak slowly (default: False)
    """
    try:
        # Print what we're about to say
        print(f"TTS: '{text}'")
        
        # Generate speech using gTTS
        tts = gTTS(text=text, lang=lang, slow=slow)
        
        # Save to a BytesIO object instead of a file
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)  # Move to the start of the BytesIO buffer
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Load the sound from the BytesIO object
        pygame.mixer.music.load(fp)
        
        # Play the sound
        pygame.mixer.music.play()
        
        # Wait for the sound to finish playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        # Clean up
        pygame.mixer.quit()
        
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Text-to-Speech utility")
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("--lang", default="en", help="Language code (e.g., 'en', 'es', 'fr')")
    parser.add_argument("--slow", action="store_true", help="Speak slowly")
    
    args = parser.parse_args()
    
    success = speak(args.text, args.lang, args.slow)
    sys.exit(0 if success else 1)