"""Commande « sentiment » : sentiment "texte" [--json]."""
import argparse
import json
import sys

from sentiment_app import __version__, predict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sentiment", description="Classe le sentiment d'un avis en français."
    )
    parser.add_argument("texte", nargs="+", help="l'avis à classer")
    parser.add_argument("--json", action="store_true", help="sortie JSON")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    args = parser.parse_args(argv)
    try:
        r = predict(" ".join(args.texte))
    except ValueError as e:
        print(f"erreur : {e}", file=sys.stderr)
        return 2
    print(
        json.dumps(r, ensure_ascii=False)
        if args.json
        else f"{r['label']} (confiance {r['score']:.0%})"
    )
    return 0
