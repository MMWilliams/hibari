"""hibari: a 7.9 MB American English text-to-speech model that runs 100x realtime on one CPU core."""
from .tts import Hibari, SAMPLE_RATE, write_wav

__version__ = "0.1.0"
__all__ = ["Hibari", "SAMPLE_RATE", "write_wav", "__version__"]
