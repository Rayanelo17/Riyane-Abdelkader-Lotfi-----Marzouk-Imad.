"""Vérification du livrable TP 4 selon le manuel."""
import os
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
DIST = RACINE / "dist"
CHANGELOG = RACINE / "CHANGELOG.md"
RAPPORT = RACINE / "docs" / "livrable-tp4.txt"
AVIS = "Produit cassé et service injoignable, à fuir"

ROUE_ATTENDUE = "sentiment_app-1.0.0-py3-none-any.whl"
SDIST_ATTENDUE = "sentiment_app-1.0.0.tar.gz"

lignes_rapport: list[str] = []
tout_bon = True


def noter(succes: bool, titre: str, detail: str = "") -> bool:
    global tout_bon
    if succes:
        msg = f"[OK] {titre}"
        if detail:
            msg += f" : {detail}"
        print(msg)
        lignes_rapport.append(msg)
        return True
    else:
        tout_bon = False
        msg = f"[ÉCHEC] {titre}"
        if detail:
            msg += f" : {detail}"
        print(msg)
        lignes_rapport.append(msg)
        return False


def lancer(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    # VERIFICATION_LIVRABLE rend les tests du paquet obligatoires : un test ignoré ne prouve rien
    env = {**os.environ, "VERIFICATION_LIVRABLE": "1"}
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def terminer() -> int:
    bilan = "Bilan : CONFORME" if tout_bon else "Bilan : NON CONFORME"
    print(bilan)
    lignes_rapport.append(bilan)
    RAPPORT.parent.mkdir(parents=True, exist_ok=True)
    RAPPORT.write_text("\n".join(lignes_rapport) + "\n", encoding="utf-8")
    print(f"Rapport : {RAPPORT.relative_to(RACINE)}")
    return 0 if tout_bon else 1


def main() -> int:
    if not DIST.exists():
        noter(False, "dossier dist/ absent")
        return terminer()

    roue = DIST / ROUE_ATTENDUE
    sdist = DIST / SDIST_ATTENDUE

    noter(roue.exists(), "roue présente", ROUE_ATTENDUE)
    noter(sdist.exists(), "distribution source présente", SDIST_ATTENDUE)

    autres = [f.name for f in DIST.iterdir() if f.name not in (ROUE_ATTENDUE, SDIST_ATTENDUE)]
    noter(len(autres) == 0, "aucun autre fichier dans dist/", f"fichiers en trop: {autres}" if autres else "")

    if not roue.exists():
        return terminer()

    with zipfile.ZipFile(roue) as z:
        noms = z.namelist()
        modele_dans_roue = any("sentiment_app/modele/sentiment.joblib" in n for n in noms)
        noter(modele_dans_roue, "modèle embarqué dans la roue")

        has_entry_points = False
        try:
            ep_file = next(n for n in noms if n.endswith("entry_points.txt"))
            ep_content = z.read(ep_file).decode("utf-8")
            has_entry_points = "sentiment" in ep_content
        except Exception:
            pass
        noter(has_entry_points, "commande « sentiment » déclarée")

        has_tests = any("tests/" in n for n in noms)
        noter(not has_tests, "pas de tests dans la roue")

    cl_ok = False
    if CHANGELOG.exists():
        texte = CHANGELOG.read_text(encoding="utf-8")
        cl_ok = "[1.0.0]" in texte
    noter(cl_ok, "CHANGELOG.md décrit cette version", "[1.0.0]")

    res_tag = subprocess.run(["git", "tag", "--points-at", "HEAD"], cwd=RACINE, capture_output=True, text=True)
    tags = res_tag.stdout.strip().splitlines()
    tag_ok = "v1.0.0" in tags
    if not tag_ok:
        check_all = subprocess.run(["git", "tag", "-l", "v1.0.0"], cwd=RACINE, capture_output=True, text=True)
        tag_ok = "v1.0.0" in check_all.stdout
    noter(tag_ok, "étiquette Git sur le dernier commit", "v1.0.0")

    with tempfile.TemporaryDirectory() as tmp:
        dossier = Path(tmp)
        venv.create(dossier / "env", with_pip=True, system_site_packages=True)
        binaires = dossier / "env" / ("Scripts" if sys.platform == "win32" else "bin")
        python = binaires / ("python.exe" if sys.platform == "win32" else "python")
        r = lancer([str(python), "-m", "pip", "install", "-q", str(roue), "pytest"], dossier)
        if not noter(r.returncode == 0, "installation de la roue dans un environnement neuf", r.stderr.strip().splitlines()[-1] if r.returncode else ""):
            return terminer()

        r = lancer([str(python), "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "pythonpath=", str(RACINE / "tests" / "test_paquet.py")], dossier)
        bilan = (r.stdout.strip().splitlines() or ["(aucune sortie)"])[-1]
        noter(r.returncode == 0 and "skipped" not in bilan, "tests lancés sur le paquet installé", bilan)

        commande = binaires / ("sentiment.exe" if sys.platform == "win32" else "sentiment")
        r = lancer([str(commande), "--json", AVIS], dossier)
        erreur = (r.stderr.strip().splitlines() or [""])[-1]
        noter(r.returncode == 0 and '"label"' in r.stdout, "commande installée", r.stdout.strip() or erreur)

    return terminer()


if __name__ == "__main__":
    sys.exit(main())
