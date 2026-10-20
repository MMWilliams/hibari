"""hibari: 7.9 MB English text-to-speech models (American and British voices) that run 100x realtime on one CPU core."""
from .tts import Hibari, SAMPLE_RATE, write_wav

__version__ = "0.2.1"
__all__ = ["Hibari", "SAMPLE_RATE", "write_wav", "__version__"]
