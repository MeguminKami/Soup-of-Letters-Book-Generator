from datetime import datetime
import json
import os
import random
from math import floor
from typing import List, Tuple, Optional, Dict

from scipy.constants import milli

ROWS, COLS = 14, 11

def export_battle(gridA, gridB, wordsA, wordsB):

    def format_grid(grid):
        return ["  " + " & ".join(str(ch) for ch in row) + " \\\\" for row in grid]

    def format_words(words):
        n = len(words)
        left_len = (n + 1) // 2
        left = words[:left_len]
        right = words[left_len:]
        right += [""] * (left_len - len(right))
        max_left = max((len(w) for w in left), default=0)
        return [f"    {left[i].ljust(max_left)} & {right[i]} \\\\" for i in range(left_len)]

    parts = []
    parts.append(r"\newcommand{\GridAcontent}{%")
    parts.append(r"  \renewcommand{\arraystretch}{1.12}%")
    parts.append(r"  \begin{tabular}{*{11}{C{0.48cm}}}")
    parts.extend(format_grid(gridA))
    parts.append(r"  \end{tabular}%")
    parts.append(r"}")
    parts.append(r"\newcommand{\GridBcontent}{%")
    parts.append(r"  \renewcommand{\arraystretch}{1.12}%")
    parts.append(r"  \begin{tabular}{*{11}{C{0.48cm}}}")
    parts.extend(format_grid(gridB))
    parts.append(r"  \end{tabular}%")
    parts.append(r"}")
    parts.append("")
    parts.append(r"\cabecalho")
    parts.append(r"\PrepararGrelhas")
    parts.append(r"\vspace{\SopaTopGap}")
    parts.append("")
    parts.append(r"\noindent")
    parts.append(r"\begin{tabular}[t]{@{}p{0.44\linewidth}@{\hspace{2mm}}p{0.54\linewidth}@{}}")
    parts.append(r"\begin{minipage}[t][\GridHeightA][t]{\linewidth}")
    parts.append(r"  \vspace*{0pt}")
    parts.append(r"  \renewcommand{\arraystretch}{1.6}")
    parts.append(r"  {\setlength{\tabcolsep}{2mm}")
    parts.append(r"  \begin{tabular}{@{}p{.48\linewidth} p{.48\linewidth}@{}}")
    parts.extend(format_words(wordsA))
    parts.append(r"  \end{tabular}")
    parts.append(r"  }")
    parts.append(r"  \\ \ \\")
    parts.append(r"  \\ \ \\")
    parts.append(r"  \campotempo")
    parts.append(r"  \vfill")
    parts.append(r"\end{minipage}")
    parts.append(r"&")
    parts.append(r"\begin{minipage}[t][\GridHeightA][t]{\linewidth}")
    parts.append(r"  \centering")
    parts.append(r"  \usebox{\GridBoxA}")
    parts.append(r"\end{minipage}\\")
    parts.append(r"\end{tabular}")
    parts.append("")
    parts.append(r"\vspace{6mm}")
    parts.append("")
    parts.append(r"\noindent")
    parts.append(r"\begin{tabular}[t]{@{}p{0.54\linewidth}@{\hspace{2mm}}p{0.44\linewidth}@{}}")
    parts.append(r"\begin{minipage}[t][\GridHeightB][t]{\linewidth}")
    parts.append(r"  \centering")
    parts.append(r"  \usebox{\GridBoxB}")
    parts.append(r"\end{minipage}")
    parts.append(r"&")
    parts.append(r"\begin{minipage}[t][\GridHeightB][t]{\linewidth}")
    parts.append(r"  \vspace*{0pt}")
    parts.append(r"  \renewcommand{\arraystretch}{1.6}")
    parts.append(r"  {\setlength{\tabcolsep}{2mm}")
    parts.append(r"  \begin{tabular}{@{}p{.48\linewidth} p{.48\linewidth}@{}}")
    parts.extend(format_words(wordsB))
    parts.append(r"  \end{tabular}")
    parts.append(r"  }")
    parts.append(r"  \\ \ \\")
    parts.append(r"  \\ \ \\")
    parts.append(r"  \campotempo")
    parts.append(r"  \vfill")
    parts.append(r"\end{minipage}\\")
    parts.append(r"\end{tabular}")
    parts.append("")
    parts.append(r"\vfill")
    parts.append(r"\newpage")
    content_text = "\n".join(parts)

    json_path = os.path.join(".", "palavras.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    counter = data.get("counter", 0)
    counter += 1
    data["counter"] = counter
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    filename = f"{counter}.tex"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content_text)


def generate_battle_words() -> Tuple[List[str], List[str]]:

    json_path = os.path.join(".", "palavras.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError("Ficheiro 'palavras.json' não encontrado no diretório atual.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_by_len: Dict[int, List[str]] = {
        4: data.get("4_letras", []),
        5: data.get("5_letras", []),
        6: data.get("6_letras", []),
        7: data.get("7_letras", []),
        8: data.get("8_letras", []),
    }

    pool: Dict[int, List[str]] = {}
    for L, words in raw_by_len.items():
        up = [w.strip().upper() for w in words if isinstance(w, str) and w.strip()]
        seen = set()
        uniq = []
        for w in up:
            if w not in seen:
                uniq.append(w); seen.add(w)
        random.shuffle(uniq)
        pool[L] = uniq

    cap = {L: floor(len(pool[L]) / 2) for L in (4,5,6,7,8)}

    rng8 = [x for x in range(1, 3+1) if x <= cap[8]]
    rng7 = [x for x in range(1, 3+1) if x <= cap[7]]
    rng6 = [x for x in range(3, 5+1) if x <= cap[6]]
    rng4 = [x for x in range(1, 3+1) if x <= cap[4]]
    attempts = 1000
    chosen = None

    for _ in range(attempts):
        c8 = random.choice(rng8)
        c7 = random.choice(rng7)
        c6 = random.choice(rng6)
        c4 = random.choice(rng4)
        c5 = 14 - (c8 + c7 + c6 + c4)

    wordsA: List[str] = []
    wordsB: List[str] = []
    for L, cL in ((8,c8),(7,c7),(6,c6),(5,c5),(4,c4)):
        if cL == 0:
            continue
        avail = pool[L]
        need = 2 * cL
        if len(avail) < need:
            raise ValueError(f"Inventário insuficiente para {L} letras: precisa {need}, tem {len(avail)}.")
        pick = random.sample(avail, need)
        random.shuffle(pick)
        wordsA.extend(pick[:cL])
        wordsB.extend(pick[cL:2*cL])

    overlap = set(wordsA) & set(wordsB)
    if overlap:
        return generate_battle_words()

    return wordsA, wordsB


def _empty_grid() -> List[List[Optional[str]]]:
    return [[None for _ in range(COLS)] for _ in range(ROWS)]


def _can_place(grid, word, r, c, dr, dc) -> bool:
    R, C = ROWS, COLS
    n = len(word)
    rr, cc = r, c
    for i in range(n):
        if not (0 <= rr < R and 0 <= cc < C):
            return False
        ch = grid[rr][cc]
        if ch is not None and ch != word[i]:
            return False
        rr += dr
        cc += dc
    return True


def _place(grid, word, r, c, dr, dc):
    rr, cc = r, c
    for ch in word:
        grid[rr][cc] = ch
        rr += dr
        cc += dc


def generate_grid_from_words(words: List[str]) -> List[List[str]]:

    words = [w.strip().upper() for w in words if isinstance(w, str) and w.strip()]

    words_sorted = sorted(words, key=len, reverse=True)

    for _attempt in range(60):
        grid = _empty_grid()
        ok = True
        for w in words_sorted:
            placed = False
            # tentativas para esta palavra
            for _ in range(400):
                horizontal = random.choice([True, False])             # coin flip 1
                w_try = w[::-1] if random.choice([True, False]) else w # coin flip 2
                if horizontal:
                    max_r = ROWS - 1
                    max_c = COLS - len(w_try)
                    if max_c < 0:
                        continue
                    r = random.randint(0, max_r)
                    c = random.randint(0, max_c)
                    dr, dc = 0, 1
                else:
                    max_r = ROWS - len(w_try)
                    max_c = COLS - 1
                    if max_r < 0:
                        continue
                    r = random.randint(0, max_r)
                    c = random.randint(0, max_c)
                    dr, dc = 1, 0
                if _can_place(grid, w_try, r, c, dr, dc):
                    _place(grid, w_try, r, c, dr, dc)
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if ok:
            # rifas
            tickets: Dict[str, int] = {ch: 1 for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
            for w in words_sorted:
                for ch in w:
                    if ch not in tickets:
                        tickets[ch] = 3   # nova letra (ex.: acentos) começa com 3
                    else:
                        tickets[ch] += 1  # letra vista ganha +1
            for v in "AEIOU":
                if v in tickets:
                    tickets[v] = max(1, round(tickets[v] * 0.6))

            letters = list(tickets.keys())
            weights = [tickets[ch] for ch in letters]

            # preenche vazios
            for r in range(ROWS):
                for c in range(COLS):
                    if grid[r][c] is None:
                        grid[r][c] = random.choices(letters, weights=weights, k=1)[0]
            return grid


if __name__ == "__main__":
    for _ in range(30):
        wordsA, wordsB = generate_battle_words()
        gridA = generate_grid_from_words(wordsA)
        gridB = generate_grid_from_words(wordsB)
        export_battle(gridA, gridB, wordsA, wordsB)
