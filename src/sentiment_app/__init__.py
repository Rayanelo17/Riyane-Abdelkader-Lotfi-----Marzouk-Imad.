"""sentiment_app : classer le sentiment d'un avis en français."""
from importlib.metadata import version

from sentiment_app.predict import predict

try:
    __version__ = version("sentiment-app")
except Exception:
    __version__ = "1.0.0"

__all__ = ["__version__", "predict"]
