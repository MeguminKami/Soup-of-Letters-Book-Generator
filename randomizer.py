import os
import re
import random
import uuid
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog


NUMBER_TEX_RE = re.compile(r"^(\d+)\.tex$", re.IGNORECASE)


def pick_folder() -> Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select folder containing numbered .tex files")
    if not folder:
        raise SystemExit("No folder selected. Exiting.")
    return Path(folder)


def find_numbered_tex_files(folder: Path):
    files = []
    for p in folder.iterdir():
        if p.is_file():
            m = NUMBER_TEX_RE.match(p.name)
            if m:
                num = int(m.group(1))
                files.append((num, p))
    files.sort(key=lambda x: x[0])
    return files


def shuffle_renames(files, seed=None):
    """
    files: list of (num, Path)
    returns mapping: old_path -> new_path
    """
    old_nums = [num for num, _ in files]
    paths = [p for _, p in files]

    new_nums = old_nums.copy()
    rng = random.Random(seed)
    rng.shuffle(new_nums)

    mapping = {}
    for old_path, new_num in zip(paths, new_nums):
        new_path = old_path.with_name(f"{new_num}.tex")
        mapping[old_path] = new_path
    return mapping


def write_mapping_file(folder: Path, mapping: dict):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = folder / f"shuffle_mapping_{ts}.txt"
    with out.open("w", encoding="utf-8") as f:
        f.write("OLD_PATH\tNEW_PATH\n")
        for oldp, newp in mapping.items():
            f.write(f"{oldp.name}\t{newp.name}\n")
    return out


def apply_renames(mapping: dict, dry_run=True):
    """
    Two-phase rename to avoid collisions:
      1) rename each file to a unique temp name
      2) rename temp names to final names
    """
    # Phase 1: temp names
    temp_map = {}
    for oldp, newp in mapping.items():
        # Keep same extension, add unique token
        tmp_name = f".__tmp__{uuid.uuid4().hex}__.tex"
        tmp_path = oldp.with_name(tmp_name)
        temp_map[oldp] = tmp_path

    if dry_run:
        print("\n[DRY RUN] Planned renames:")
        for oldp, newp in mapping.items():
            print(f"  {oldp.name}  ->  {newp.name}")
        print("\nNo files were renamed (dry run).")
        return

    # Actually rename old -> temp
    for oldp, tmpp in temp_map.items():
        os.replace(oldp, tmpp)

    # Rename temp -> final
    for oldp, newp in mapping.items():
        tmpp = temp_map[oldp]
        os.replace(tmpp, newp)


def main():
    folder = pick_folder()
    files = find_numbered_tex_files(folder)

    if not files:
        print(f"No files matching NUMBER.tex found in: {folder}")
        return

    print(f"Found {len(files)} numbered .tex files in: {folder}")
    # Options:
    DRY_RUN = True          # change to False to actually rename
    SEED = None             # set e.g. 123 for repeatable shuffles

    mapping = shuffle_renames(files, seed=SEED)
    mapping_file = write_mapping_file(folder, mapping)
    print(f"Mapping saved to: {mapping_file}")

    apply_renames(mapping, dry_run=DRY_RUN)
    if DRY_RUN:
        print("\nSet DRY_RUN = False in the script to apply the renames.")


if __name__ == "__main__":
    main()
