"""Mini démonstration d'argparse.

Exemples :
    python utils/argparse_demo.py Alice
    python utils/argparse_demo.py Alice --age 12 --loud
    python utils/argparse_demo.py --help
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mini programme pour comprendre argparse."
    )

    # Argument positionnel : sa valeur est attendue sans écrire --name.
    parser.add_argument(
        "name",
        help="Nom de la personne.",
    )

    # Option nommée avec une valeur : --age 12.
    parser.add_argument(
        "--age",
        type=int,
        default=None,
        help="Âge facultatif.",
    )

    # Option booléenne : sa présence suffit à la mettre à True.
    parser.add_argument(
        "--loud",
        action="store_true",
        help="Affiche le message en majuscules.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    message = f"Bonjour {args.name}"
    if args.age is not None:
        message += f", tu as {args.age} ans"
    message += " !"

    if args.loud:
        message = message.upper()

    print(message)
    print(f"[debug] args.name = {args.name!r}")
    print(f"[debug] args.age = {args.age!r}")
    print(f"[debug] args.loud = {args.loud!r}")


if __name__ == "__main__":
    main()
