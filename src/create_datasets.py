import click

from pathlib import Path

from preprocessing import (
    create_numbers_dataset,
    create_lowercase_latin_letters_dataset,
    create_lowercase_latin_letters_with_diacritics_dataset,
    create_latin_letters_dataset,
    create_latin_letters_with_diacritics_dataset,
    create_edge_case_dataset,
)


@click.command()
def main():
    """Creates datasets"""
    print("Creating datasets...")

    create_numbers_dataset()
    # create_lowercase_latin_letters_dataset()
    # create_lowercase_latin_letters_with_diacritics_dataset()
    # create_latin_letters_dataset()
    # create_latin_letters_with_diacritics_dataset()
    # create_edge_case_dataset()


if __name__ == "__main__":
    main()
