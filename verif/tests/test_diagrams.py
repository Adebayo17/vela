"""
test_diagrams.py — The SVG diagrams of docs/diagrams/ are valid and up to date

Chaque schéma est généré par tools/gen_diagrams.py. On vérifie :
  - que c'est du XML bien formé (un attribut en double, par exemple, casse
    l'affichage sur GitHub alors qu'un navigateur le tolère) ;
  - que le fichier du dépôt est identique à ce que produit le script.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from vela_testing import requirement

import gen_diagrams

DOCS = Path(__file__).resolve().parents[2] / "docs" / "diagrams"
NAMES = sorted(gen_diagrams.DIAGRAMS)


@requirement("VELA-FR-007")
@pytest.mark.parametrize("name", NAMES)
def test_diagram_is_valid_xml(name):
    """Le SVG généré est du XML bien formé, avec un titre accessible."""
    root = ET.fromstring(gen_diagrams.DIAGRAMS[name]())
    assert root.tag.endswith("svg")
    assert root.find("{http://www.w3.org/2000/svg}title") is not None


@requirement("VELA-FR-007")
@pytest.mark.parametrize("name", NAMES)
def test_diagram_up_to_date(name):
    """docs/diagrams/<name> = sortie actuelle du générateur (sinon : make diagrams)."""
    assert (DOCS / name).read_text() == gen_diagrams.DIAGRAMS[name](), \
        f"{name} is stale: run `make diagrams`"
