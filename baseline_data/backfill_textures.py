"""
osvi-awda's own .gitignore has a blanket `*.png` rule, so every MuJoCo texture PNG referenced by
the vendored robosuite/hem XML assets was never actually committed to this repo - not corruption,
just a structural gap discovered while smoke-testing collect_demonstrations.py (it crashed trying
to mj_loadXML the bin arena's 'tex-light-wood' texture, which turned out not to exist on disk at
all). This script finds every <texture file="..."> reference under the directories relevant to
PandaPickPlaceDistractor/SawyerPickPlaceDistractor (hem/robosuite/, robosuite/models/) and:
  1. Copies a same-named file from another installed robosuite package on this cluster where one
     exists (covers the ~15 standard robosuite textures: light-wood, dark-wood, bread, can,
     ceramic, cereal, glass, lemon, metal, ...).
  2. Generates a simple solid-color 64x64 PNG placeholder for the rest (~26 files) - these are all
     branded distractor-object textures (coke, cheezeits, tictac, ...) unique to this repo with no
     legitimate copy available anywhere; purely cosmetic (physics/success signals come from
     position/contact, not texture appearance), so a solid color per object is an acceptable
     placeholder rather than leaving the sim unable to load at all.

Idempotent - only touches files that don't already exist. Re-run this if the repo is re-cloned
fresh (since .gitignore's *.png rule means these generated files won't survive a commit either -
this is a local, one-time environment-setup step, not a permanent fix to the repo's asset
tracking, which is out of scope for this task).

Run with any python3 that has Pillow (does not need the 'awda' env - no torch/mujoco_py import).
"""
import os
import re
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH_DIRS = [os.path.join(REPO_ROOT, "hem", "robosuite"),
               os.path.join(REPO_ROOT, "robosuite", "models")]
# Other robosuite installs on this cluster that happen to ship the same standard texture names.
SOURCE_DIRS = [
    "/mnt/beegfs/frosa/.conda/envs/openvla_robosuite_1_0_1/lib/python3.9/site-packages/robosuite/models/assets/textures",
    "/mnt/beegfs/frosa/.conda/envs/multi_task_robosuite_1_5/lib/python3.10/site-packages/robosuite/models/assets/textures",
    "/mnt/beegfs/frosa/.conda/envs/libero_pro/lib/python3.8/site-packages/robosuite/models/assets/textures",
]
COLOR_KEYWORDS = {
    "coke": (139, 0, 0), "sprite": (34, 139, 34), "fanta": (255, 140, 0),
    "mdew": (154, 205, 50), "drpepper": (101, 67, 33),
    "cheezeits": (255, 165, 0), "frito": (218, 165, 32), "bounce": (240, 230, 140),
    "clean": (70, 130, 180), "woolite": (255, 192, 203),
    "banana": (255, 225, 53), "orange": (255, 140, 0), "pear": (163, 209, 91),
    "altoid": (192, 192, 192), "candy": (220, 20, 60), "cardboard": (181, 141, 90),
    "claratin": (255, 255, 255), "dove": (240, 248, 255), "five": (0, 0, 128),
    "motrin": (220, 20, 60), "tictac": (255, 255, 255), "zertec": (255, 255, 255),
    "dutchchips": (255, 215, 0), "purplepringle": (128, 0, 128), "redpringle": (200, 0, 0),
    "whiteclaw": (200, 200, 200),
}


def color_for(basename):
    stem = os.path.splitext(basename)[0].lower()
    for kw, color in COLOR_KEYWORDS.items():
        if kw in stem:
            return color
    return (160, 160, 160)


def main():
    pat = re.compile(r'texture[^>]*\bfile="([^"]+)"')
    needed = set()
    for d in SEARCH_DIRS:
        for root, _, files in os.walk(d):
            for f in files:
                if not f.endswith(".xml"):
                    continue
                with open(os.path.join(root, f), errors="ignore") as fh:
                    content = fh.read()
                for rel in pat.findall(content):
                    if rel.startswith("/Users/") or rel.startswith("/home/"):
                        # hardcoded absolute path from the original authors' machines (seen in
                        # robosuite/models/assets/demonstrations/*), not part of the live
                        # Panda/Sawyer PickPlaceDistractor env-construction path - skip.
                        continue
                    needed.add(os.path.normpath(os.path.join(root, rel)))

    missing = sorted(t for t in needed if not os.path.exists(t))
    print(f"total texture refs: {len(needed)}, missing: {len(missing)}")
    if not missing:
        print("nothing to do.")
        return

    from PIL import Image
    filled, placeholders = [], []
    for target in missing:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        basename = os.path.basename(target)
        src = next((os.path.join(sd, basename) for sd in SOURCE_DIRS
                    if os.path.exists(os.path.join(sd, basename))), None)
        if src:
            shutil.copy(src, target)
            filled.append(target)
        else:
            Image.new("RGB", (64, 64), color_for(basename)).save(target)
            placeholders.append(target)

    print(f"copied from a legitimate source: {len(filled)}")
    print(f"generated solid-color placeholders: {len(placeholders)}")
    for p in placeholders:
        print("  placeholder:", p)


if __name__ == "__main__":
    main()
