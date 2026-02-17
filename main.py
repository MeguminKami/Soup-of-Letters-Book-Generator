from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from colorama import Fore, Back, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    colorama_init = None  # type: ignore
    # Fallback to empty strings if colorama is not available
    class _DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = _DummyColor()
    Back = _DummyColor()
    Style = _DummyColor()


ASCII_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BASE_LENGTH_RATIO: Dict[int, int] = {4: 2, 5: 4, 6: 4, 7: 2, 8: 2}
SUPPORTED_LENGTHS = tuple(sorted(BASE_LENGTH_RATIO.keys()))
GRID_STYLE_BY_COLS: Dict[int, Tuple[float, str, float]] = {
    11: (0.62, r"\Large", 1.18),
    12: (0.56, r"\large", 1.14),  # Type F - tall format (now 30x12)
    14: (0.50, r"\large", 1.12),  # Type E - battle format (14x14)
    15: (0.50, r"\large", 1.12),  # Reserved
    17: (0.46, r"\normalsize", 1.10),
    20: (0.52, r"\large", 1.15),  # Type D - larger cells and spacing
    26: (0.39, r"\footnotesize", 1.02),
}
FOLDER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._\\/-]+$")
BACK_COMMANDS = {"B", "BACK", "VOLTAR"}
QUIT_COMMANDS = {"Q", "QUIT", "EXIT", "SAIR"}
NAV_BACK = "__BACK__"
NAV_QUIT = "__QUIT__"


# Terminal Display Functions
def clear_screen() -> None:
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(text: str) -> None:
    """Print a formatted header with colors."""
    width = 70
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'═' * width}")
    print(f"{text.center(width)}")
    print(f"{'═' * width}{Style.RESET_ALL}\n")


def print_section(text: str) -> None:
    """Print a section title."""
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}▶ {text}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'─' * 70}{Style.RESET_ALL}")


def print_box(lines: List[str], color: str = Fore.WHITE) -> None:
    """Print text in a box."""
    if not lines:
        return
    max_len = max(len(line) for line in lines)
    width = max_len + 4

    print(f"{color}┌{'─' * (width - 2)}┐")
    for line in lines:
        padding = max_len - len(line)
        print(f"│ {line}{' ' * padding} │")
    print(f"└{'─' * (width - 2)}┘{Style.RESET_ALL}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Fore.GREEN}{Style.BRIGHT}✓ {text}{Style.RESET_ALL}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Fore.RED}{Style.BRIGHT}✗ {text}{Style.RESET_ALL}")


def print_info(text: str) -> None:
    """Print info message."""
    print(f"{Fore.CYAN}ℹ {text}{Style.RESET_ALL}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")


@dataclass(frozen=True)
class DifficultyConfig:
    key: str
    label: str
    color_macro: str
    bonus: int


DIFFICULTIES: Dict[str, DifficultyConfig] = {
    "1": DifficultyConfig(key="easy", label="Fácil", color_macro="difficultyEasy", bonus=0),
    "2": DifficultyConfig(key="medium", label="Médio", color_macro="difficultyMedium", bonus=1),
    "3": DifficultyConfig(key="hard", label="Dificil", color_macro="difficultyHard", bonus=3),
    "4": DifficultyConfig(key="super_hard", label="Impossível", color_macro="difficultySuperHard", bonus=5),
}


class WordProvider:
    def sample_words(self, length_plan: Sequence[int], used_tokens: Set[str], used_words: Sequence[str]) -> List[str]:
        raise NotImplementedError


class JsonWordProvider(WordProvider):
    def __init__(self, pools: Dict[int, List[str]], rng: random.Random):
        self._pools = {length: list(words) for length, words in pools.items()}
        self._rng = rng
        for words in self._pools.values():
            self._rng.shuffle(words)

    def sample_words(self, length_plan: Sequence[int], used_tokens: Set[str], used_words: Sequence[str]) -> List[str]:
        needed = Counter(length_plan)
        selected: List[str] = []
        local_tokens = set(used_tokens)

        for length, count in needed.items():
            candidates = list(self._pools.get(length, []))
            self._rng.shuffle(candidates)
            picked = 0
            for word in candidates:
                if has_stem_conflict(word, local_tokens):
                    continue
                selected.append(word)
                local_tokens.update(stem_tokens_for_word(word))
                picked += 1
                if picked == count:
                    break
            if picked < count:
                raise RuntimeError(
                    f"Not enough unique words with {length} letters for this request "
                    "(consider reducing pages or using AI mode)."
                )

        self._rng.shuffle(selected)
        return selected


class AIWordProvider(WordProvider):
    def __init__(self, themes: Sequence[str], rng: random.Random):
        self._themes = list(themes)
        self._rng = rng
        # Import and initialize the AI generator
        try:
            from ai import GeradorPalavrasIA
            self._generator = GeradorPalavrasIA()
        except ImportError:
            raise RuntimeError("Modo IA requer o módulo ai.py com a classe GeradorPalavrasIA")

        self._page_counter = 0

    def sample_words(self, length_plan: Sequence[int], used_tokens: Set[str], used_words: Sequence[str]) -> List[str]:
        self._page_counter += 1

        # Convert used_words to set for faster lookup and include all used tokens
        all_used = set(used_words) | used_tokens

        # Call the AI generator with proper parameters
        words = self._generator.gerar_palavras(
            quantidade=len(length_plan),
            plano_comprimentos=list(length_plan),
            temas=self._themes,
            palavras_usadas=all_used,
            numero_pagina=self._page_counter
        )

        # Shuffle for variety
        self._rng.shuffle(words)
        return words




# Old AI helper functions - now handled in ai.py module
# These are kept commented for reference but no longer used

# def build_ai_client():
#     ...existing code...

# def strip_code_fences(text: str) -> str:
#     ...existing code...

# def extract_json_payload(text: str) -> str:
#     ...existing code...

# def parse_and_validate_ai_words(...):
#     ...existing code...

# def normalize_word(raw_word: str) -> str:
#     ...existing code...

# def stem_tokens_for_word(word: str) -> Set[str]:
#     ...existing code...

# def has_stem_conflict(word: str, used_tokens: Set[str]) -> bool:
#     ...existing code...


def load_word_pools(palavras_dir: Path) -> Dict[int, List[str]]:
    pools: Dict[int, List[str]] = {}
    for length in SUPPORTED_LENGTHS:
        file_path = palavras_dir / f"{length}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing words file: {file_path}")

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        raw_words = data.get("words")
        if not isinstance(raw_words, list):
            raise ValueError(f"Invalid format in {file_path}: expected key 'words' with a list")

        seen: Set[str] = set()
        cleaned: List[str] = []
        for raw in raw_words:
            if not isinstance(raw, str):
                continue
            word = normalize_word(raw)
            if len(word) != length:
                continue
            if word not in seen:
                seen.add(word)
                cleaned.append(word)

        if not cleaned:
            raise ValueError(f"No usable words with length {length} in {file_path}")
        pools[length] = cleaned
    return pools


def build_length_plan(total_words: int, rng: random.Random) -> List[int]:
    total_weight = sum(BASE_LENGTH_RATIO.values())
    raw = {length: (BASE_LENGTH_RATIO[length] * total_words) / total_weight for length in SUPPORTED_LENGTHS}
    counts = {length: int(raw[length]) for length in SUPPORTED_LENGTHS}
    remaining = total_words - sum(counts.values())

    if remaining > 0:
        ranked = sorted(
            SUPPORTED_LENGTHS,
            key=lambda length: (raw[length] - counts[length], rng.random()),
            reverse=True,
        )
        for index in range(remaining):
            counts[ranked[index % len(ranked)]] += 1

    plan: List[int] = []
    for length in SUPPORTED_LENGTHS:
        plan.extend([length] * counts[length])
    rng.shuffle(plan)
    return plan


def empty_grid(rows: int, cols: int) -> List[List[str | None]]:
    return [[None for _ in range(cols)] for _ in range(rows)]


def can_place(
    grid: List[List[str | None]],
    word: str,
    start_row: int,
    start_col: int,
    step_row: int,
    step_col: int,
) -> bool:
    rows = len(grid)
    cols = len(grid[0])
    row, col = start_row, start_col
    for index in range(len(word)):
        if not (0 <= row < rows and 0 <= col < cols):
            return False
        current = grid[row][col]
        if current is not None and current != word[index]:
            return False
        row += step_row
        col += step_col
    return True


def place_word(
    grid: List[List[str | None]],
    word: str,
    start_row: int,
    start_col: int,
    step_row: int,
    step_col: int,
) -> None:
    row, col = start_row, start_col
    for char in word:
        grid[row][col] = char
        row += step_row
        col += step_col


def build_fill_tickets(words: Sequence[str], difficulty_bonus: int) -> Dict[str, int]:
    tickets: Dict[str, int] = {char: 1 for char in ASCII_UPPER}
    increment = 1 + difficulty_bonus
    for word in words:
        for char in word:
            tickets[char] = tickets.get(char, 1) + increment

    for vowel in "AEIOU":
        tickets[vowel] = max(1, round(tickets[vowel] * 0.6))
    return tickets


def generate_grid(words: Sequence[str], rows: int, cols: int, difficulty_bonus: int, rng: random.Random) -> List[List[str]]:
    sorted_words = sorted(words, key=len, reverse=True)
    directions = ((0, 1), (1, 0))

    for _ in range(120):
        grid = empty_grid(rows, cols)
        success = True

        for word in sorted_words:
            placed = False
            for _ in range(700):
                step_row, step_col = rng.choice(directions)
                candidate = word[::-1] if rng.random() < 0.5 else word

                if step_row == 0:
                    max_row = rows - 1
                    max_col = cols - len(candidate)
                    if max_col < 0:
                        continue
                    row = rng.randint(0, max_row)
                    col = rng.randint(0, max_col)
                else:
                    max_row = rows - len(candidate)
                    max_col = cols - 1
                    if max_row < 0:
                        continue
                    row = rng.randint(0, max_row)
                    col = rng.randint(0, max_col)

                if can_place(grid, candidate, row, col, step_row, step_col):
                    place_word(grid, candidate, row, col, step_row, step_col)
                    placed = True
                    break

            if not placed:
                success = False
                break

        if not success:
            continue

        tickets = build_fill_tickets(sorted_words, difficulty_bonus)
        letters = list(tickets.keys())
        weights = [tickets[char] for char in letters]

        for row in range(rows):
            for col in range(cols):
                if grid[row][col] is None:
                    grid[row][col] = rng.choices(letters, weights=weights, k=1)[0]

        return [[char if char is not None else "A" for char in row] for row in grid]

    raise RuntimeError(f"Failed to build a {rows}x{cols} grid after many attempts.")


def indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else prefix for line in text.splitlines())


def render_grid_tabular(grid: Sequence[Sequence[str]]) -> str:
    cols = len(grid[0])
    # Use consistent formatting for battle pages (11 and 14 cols)
    if cols == 11 or cols == 14:
        cell_width = 0.48 if cols == 11 else 0.44
        lines = [
            r"\renewcommand{\arraystretch}{1.12}%",
            "\\begin{tabular}{*{%d}{C{%.2fcm}}}" % (cols, cell_width),
        ]
    elif cols == 20:
        # Type D - larger cells, more spacing
        cell_width = 0.52
        lines = [
            r"\renewcommand{\arraystretch}{1.15}",
            r"\setlength{\tabcolsep}{1.5pt}",
            r"\large",
            "\\begin{tabular}{*{%d}{C{%.2fcm}}}" % (cols, cell_width),
        ]
    else:
        # For other grid sizes, use original logic
        cell_width, font_cmd, row_stretch = GRID_STYLE_BY_COLS.get(cols, (0.30, r"\normalsize", 1.10))
        lines = [
            rf"\renewcommand{{\arraystretch}}{{{row_stretch:.2f}}}",
            r"\setlength{\tabcolsep}{1pt}",
            font_cmd,
            "\\begin{tabular}{*{%d}{C{%.2fcm}}}" % (cols, cell_width),
        ]

    for row in grid:
        lines.append("  " + " & ".join(row) + r" \\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def render_grid_box(grid: Sequence[Sequence[str]], grid_id: str = "") -> str:
    """Render grid. For battle pages, returns just the reference; content is in GridAcontent/GridBcontent."""
    if grid_id:
        # For battle pages, just return the usebox reference
        return f"\\usebox{{\\GridBox{grid_id}}}"
    else:
        # For single puzzle pages, return the full framed grid
        tabular = render_grid_tabular(grid)
        return "\\PuzzleFrame{%\n" + indent_block(tabular, "  ") + "\n}"


def render_two_column_word_table(words: Sequence[str]) -> str:
    # Sort by length (decreasing), then alphabetically
    sorted_words = sorted(words, key=lambda w: (-len(w), w))

    split = (len(sorted_words) + 1) // 2
    left = list(sorted_words[:split])
    right = list(sorted_words[split:])
    right.extend([""] * (len(left) - len(right)))

    lines = [
        r"\begin{tabular}{@{}p{.48\linewidth} p{.48\linewidth}@{}}",
    ]
    for index in range(len(left)):
        lines.append(f"    {left[index]} & {right[index]} \\\\")
    lines.append(r"  \end{tabular}")
    return "\n".join(lines)


def render_vertical_word_table(words: Sequence[str]) -> str:
    # Sort by length (decreasing), then alphabetically
    sorted_words = sorted(words, key=lambda w: (-len(w), w))

    lines = [
        r"\renewcommand{\arraystretch}{1.58}",
        r"\begin{tabular}{@{}l@{}}",
    ]
    for word in sorted_words:
        lines.append(f"  {word} \\\\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def render_three_column_word_table(words: Sequence[str], rows_per_col: int = 9) -> str:
    # Sort by length (decreasing), then alphabetically
    sorted_words = sorted(words, key=lambda w: (-len(w), w))

    columns: List[List[str]] = []
    for col_index in range(3):
        start = col_index * rows_per_col
        columns.append(list(sorted_words[start : start + rows_per_col]))

    lines = [
        r"\renewcommand{\arraystretch}{1.58}",
        r"\setlength{\tabcolsep}{2.4mm}",
        r"\begin{tabular}{@{}lll@{}}",
    ]
    for row_index in range(rows_per_col):
        col_a = columns[0][row_index] if row_index < len(columns[0]) else ""
        col_b = columns[1][row_index] if row_index < len(columns[1]) else ""
        col_c = columns[2][row_index] if row_index < len(columns[2]) else ""
        lines.append(f"  {col_a} & {col_b} & {col_c} \\\\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def render_six_column_word_table(words: Sequence[str], rows_per_col: int = 5) -> str:
    # Sort by length (decreasing), then alphabetically
    sorted_words = sorted(words, key=lambda w: (-len(w), w))

    columns: List[List[str]] = []
    for col_index in range(6):
        start = col_index * rows_per_col
        columns.append(list(sorted_words[start : start + rows_per_col]))

    lines = [
        r"\renewcommand{\arraystretch}{1.58}",
        r"\setlength{\tabcolsep}{1.8mm}",
        r"\begin{tabular}{@{}llllll@{}}",
    ]
    for row_index in range(rows_per_col):
        row_words = []
        for col_index in range(6):
            if row_index < len(columns[col_index]):
                row_words.append(columns[col_index][row_index])
            else:
                row_words.append("")
        lines.append(f"  {' & '.join(row_words)} \\\\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def render_battle_word_panel(words: Sequence[str], include_time: bool, grid_height: str = "A") -> str:
    table = render_two_column_word_table(words)
    lines = [
        f"\\begin{{minipage}}[t][\\GridHeight{grid_height}][t]{{\\linewidth}}",
        r"  \vspace*{0pt}",
        r"  \renewcommand{\arraystretch}{1.6}",
        r"  {\setlength{\tabcolsep}{2mm}",
        indent_block(table, "  "),
        r"  }",
    ]
    if include_time:
        lines.extend([
            r"  \\ \ \\",
            r"  \\ \ \\",
            r"  \campotempo",
            r"  \vfill"
        ])
    else:
        lines.append(r"  \vfill")
    lines.append(r"\end{minipage}")
    return "\n".join(lines)


def render_side_word_panel(words: Sequence[str]) -> str:
    table = render_vertical_word_table(words)
    lines = [
        r"\begin{minipage}[t]{\linewidth}",
        r"  \vspace*{0pt}",
        indent_block(table, "  "),
        r"\end{minipage}",
    ]
    return "\n".join(lines)


def render_top_bottom_word_panel(words: Sequence[str]) -> str:
    # Use 6 columns with ~5 rows per column for 27 words
    table = render_six_column_word_table(words, rows_per_col=5)
    lines = [
        r"\begin{minipage}[t]{\linewidth}",
        r"  \centering",
        indent_block(table, "  "),
        r"\end{minipage}",
    ]
    return "\n".join(lines)


def render_battle_page(
    grid_a: Sequence[Sequence[str]],
    grid_b: Sequence[Sequence[str]],
    words_a: Sequence[str],
    words_b: Sequence[str],
    difficulty: DifficultyConfig,
    with_details: bool,
) -> str:
    # Render grid content as newcommands
    grid_a_tabular = render_grid_tabular(grid_a)
    grid_b_tabular = render_grid_tabular(grid_b)

    lines = [
        r"\newcommand{\GridAcontent}{%",
        indent_block(grid_a_tabular, "  ") + "%",
        r"}",
        r"\newcommand{\GridBcontent}{%",
        indent_block(grid_b_tabular, "  ") + "%",
        r"}",
        r"",
    ]

    # Add header
    if with_details:
        # Type A: cabecalho (date/winner) + page number at bottom right
        lines.append(r"\cabecalho")
    else:
        # Type B: date at top left, page number at top right
        lines.extend([
            r"\noindent",
            r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}r@{}}",
            r"\textbf{Data:} \underline{\hspace{0.6cm}} /\, \underline{\hspace{0.6cm}} /\, \underline{\hspace{1.1cm}}",
            r"& \thepage \\",
            r"\end{tabular*}",
            r"\vspace{3mm}",
        ])

    lines.extend([
        r"\PrepararGrelhas",
        r"\vspace{\SopaTopGap}",
        r"",
        r"\noindent",
        r"\begin{tabular}[t]{@{}p{0.44\linewidth}@{\hspace{2mm}}p{0.54\linewidth}@{}}",
    ])

    # First puzzle (words on left, grid on right)
    words_panel_a = render_battle_word_panel(words_a, include_time=with_details, grid_height="A")
    lines.extend([
        words_panel_a,
        r"&",
        r"\begin{minipage}[t][\GridHeightA][t]{\linewidth}",
        r"  \centering",
        r"  \usebox{\GridBoxA}",
        r"\end{minipage}\\",
        r"\end{tabular}",
        r"",
        r"\vspace{6mm}",
        r"",
    ])

    # Second puzzle (grid on left, words on right)
    words_panel_b = render_battle_word_panel(words_b, include_time=with_details, grid_height="B")
    lines.extend([
        r"\noindent",
        r"\begin{tabular}[t]{@{}p{0.54\linewidth}@{\hspace{2mm}}p{0.44\linewidth}@{}}",
        r"\begin{minipage}[t][\GridHeightB][t]{\linewidth}",
        r"  \centering",
        r"  \usebox{\GridBoxB}",
        r"\end{minipage}",
        r"&",
        words_panel_b + "\\",
        r"\end{tabular}",
        r"",
        r"\vfill",
    ])

    # Add page number at bottom right for Type A
    if with_details:
        lines.extend([
            r"\noindent",
            r"\hfill\thepage",
            r"",
        ])

    lines.append(r"\newpage")

    return "\n".join(lines) + "\n"


def render_type_c_page(
    grid: Sequence[Sequence[str]],
    words: Sequence[str],
    difficulty: DifficultyConfig,
    grid_on_left: bool,
) -> str:
    ordered_words = sorted(words, key=lambda word: (-len(word), word))
    words_panel = render_side_word_panel(ordered_words)
    grid_panel = render_grid_box(grid)

    left_block = grid_panel if grid_on_left else words_panel
    right_block = words_panel if grid_on_left else grid_panel

    lines = [
        r"\noindent",
        r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}r@{}}",
        r"\textbf{Data:} \underline{\hspace{0.6cm}} /\, \underline{\hspace{0.6cm}} /\, \underline{\hspace{1.1cm}}",
        r"& \thepage \\",
        r"\end{tabular*}",
        r"\vspace{3mm}",
        r"",
        r"\begin{tabular}[t]{@{}p{0.74\linewidth}@{\hspace{2mm}}p{0.24\linewidth}@{}}",
        indent_block(left_block, "  "),
        r"  &",
        indent_block(right_block, "  "),
        r"\end{tabular}",
        r"\newpage",
    ]
    return "\n".join(lines) + "\n"


def render_type_d_page(
    grid: Sequence[Sequence[str]],
    words: Sequence[str],
    difficulty: DifficultyConfig,
    words_on_top: bool,
) -> str:
    ordered_words = sorted(words, key=lambda word: (-len(word), word))
    words_panel = render_top_bottom_word_panel(ordered_words)
    grid_panel = render_grid_box(grid)

    lines: List[str] = [
        r"\noindent",
        r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}r@{}}",
        r"\textbf{Data:} \underline{\hspace{0.6cm}} /\, \underline{\hspace{0.6cm}} /\, \underline{\hspace{1.1cm}}",
        r"& \thepage \\",
        r"\end{tabular*}",
        r"\vspace{3mm}",
        r"",
    ]
    if words_on_top:
        lines.extend(
            [
                r"\vspace{2mm}",
                words_panel,
                r"\vspace{5mm}",
                r"\begin{center}",
                grid_panel,
                r"\end{center}",
                r"\vfill",
            ]
        )
    else:
        lines.extend(
            [
                r"\vspace{2mm}",
                r"\begin{center}",
                grid_panel,
                r"\end{center}",
                r"\vspace{5mm}",
                words_panel,
                r"\vfill",
            ]
        )
    lines.append(r"\newpage")
    return "\n".join(lines) + "\n"


def render_four_column_word_table(words: Sequence[str]) -> str:
    """Render words in 4 columns for Type E."""
    # Sort by length (decreasing), then alphabetically
    sorted_words = sorted(words, key=lambda w: (-len(w), w))

    rows_per_col = (len(sorted_words) + 3) // 4
    columns: List[List[str]] = []
    for col_index in range(4):
        start = col_index * rows_per_col
        columns.append(list(sorted_words[start : start + rows_per_col]))

    lines = [
        r"\renewcommand{\arraystretch}{1.5}",
        r"\setlength{\tabcolsep}{2mm}",
        r"\footnotesize",
        r"\begin{tabular}{@{}llll@{}}",
    ]
    for row_index in range(rows_per_col):
        row_words = []
        for col_index in range(4):
            if row_index < len(columns[col_index]):
                row_words.append(columns[col_index][row_index])
            else:
                row_words.append("")
        lines.append(f"  {' & '.join(row_words)} \\\\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def render_type_e_page(
    grid_a: Sequence[Sequence[str]],
    grid_b: Sequence[Sequence[str]],
    words_a: Sequence[str],
    words_b: Sequence[str],
    difficulty: DifficultyConfig,
) -> str:
    """Type E: Battle format like Type B but with 16 words each (14+2) and 14 cols (11+3)."""
    # Render grid content as newcommands
    grid_a_tabular = render_grid_tabular(grid_a)
    grid_b_tabular = render_grid_tabular(grid_b)

    lines = [
        r"\newcommand{\GridAcontent}{%",
        indent_block(grid_a_tabular, "  ") + "%",
        r"}",
        r"\newcommand{\GridBcontent}{%",
        indent_block(grid_b_tabular, "  ") + "%",
        r"}",
        r"",
        r"\noindent",
        r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}r@{}}",
        r"\textbf{Data:} \underline{\hspace{0.6cm}} /\, \underline{\hspace{0.6cm}} /\, \underline{\hspace{1.1cm}}",
        r"& \thepage \\",
        r"\end{tabular*}",
        r"\vspace{3mm}",
        r"",
        r"\PrepararGrelhas",
        r"\vspace{\SopaTopGap}",
        r"",
        r"\noindent",
        r"\begin{tabular}[t]{@{}p{0.44\linewidth}@{\hspace{2mm}}p{0.54\linewidth}@{}}",
    ]

    # First puzzle (words on left, grid on right) - no campotempo
    words_panel_a = render_battle_word_panel(words_a, include_time=False, grid_height="A")
    lines.extend([
        words_panel_a,
        r"&",
        r"\begin{minipage}[t][\GridHeightA][t]{\linewidth}",
        r"  \centering",
        r"  \usebox{\GridBoxA}",
        r"\end{minipage}\\",
        r"\end{tabular}",
        r"",
        r"\vspace{6mm}",
        r"",
    ])

    # Second puzzle (grid on left, words on right) - no campotempo
    words_panel_b = render_battle_word_panel(words_b, include_time=False, grid_height="B")
    lines.extend([
        r"\noindent",
        r"\begin{tabular}[t]{@{}p{0.54\linewidth}@{\hspace{2mm}}p{0.44\linewidth}@{}}",
        r"\begin{minipage}[t][\GridHeightB][t]{\linewidth}",
        r"  \centering",
        r"  \usebox{\GridBoxB}",
        r"\end{minipage}",
        r"&",
        r"\hspace{8mm}" + words_panel_b + "\\",  # Offset words to the right
        r"\end{tabular}",
        r"",
        r"\vfill",
        r"\newpage",
    ])

    return "\n".join(lines) + "\n"


def render_type_f_page(
    grid: Sequence[Sequence[str]],
    words: Sequence[str],
    difficulty: DifficultyConfig,
) -> str:
    """Type F: Tall format (30x12) with 22 words on the side."""
    # Words are already sorted by caller
    words_panel = render_side_word_panel(words)
    grid_panel = render_grid_box(grid)

    lines = [
        r"\noindent",
        r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}r@{}}",
        r"\textbf{Data:} \underline{\hspace{0.6cm}} /\, \underline{\hspace{0.6cm}} /\, \underline{\hspace{1.1cm}}",
        r"& \thepage \\",
        r"\end{tabular*}",
        r"\vspace{3mm}",
        r"",
        r"\begin{tabular}[t]{@{}p{0.68\linewidth}@{\hspace{3mm}}p{0.29\linewidth}@{}}",
        r"\begin{minipage}[t]{\linewidth}",
        r"  \centering",
        indent_block(grid_panel, "  "),
        r"\end{minipage}",
        r"&",
        indent_block(words_panel, ""),
        r"\\",
        r"\end{tabular}",
        r"\vfill",
        r"\newpage",
    ]
    return "\n".join(lines) + "\n"


def register_words(words: Sequence[str], used_tokens: Set[str], used_words: List[str], used_word_set: Set[str]) -> None:
    for word in words:
        used_tokens.update(stem_tokens_for_word(word))
        if word not in used_word_set:
            used_word_set.add(word)
            used_words.append(word)


def ask_choice(
    title: str,
    options: Sequence[Tuple[str, str]],
    *,
    allow_back: bool = False,
    allow_quit: bool = False,
) -> str:
    print_section(title)

    # Display regular options
    for key, description in options:
        print(f"  {Fore.GREEN}{Style.BRIGHT}{key}{Style.RESET_ALL}) {description}")

    # Add Back as last numbered option if needed
    if allow_back:
        # Find the next available number
        existing_keys = [key for key, _ in options]
        next_num = str(len(existing_keys) + 1)
        print(f"  {Fore.YELLOW}{Style.BRIGHT}{next_num}{Style.RESET_ALL}) ← Back")
        # Create extended options list
        options_with_back = list(options) + [(next_num, "Back")]
    else:
        options_with_back = list(options)

    valid = {key.upper(): key for key, _ in options_with_back}
    back_inputs = set(BACK_COMMANDS)
    quit_inputs = set(QUIT_COMMANDS)

    if "B" in valid:
        back_inputs.discard("B")
    if "Q" in valid:
        quit_inputs.discard("Q")

    while True:
        choice = input(f"\n{Fore.CYAN}➤ Your choice: {Style.RESET_ALL}").strip().upper()
        if allow_back and choice in back_inputs:
            return NAV_BACK
        if allow_quit and choice in quit_inputs:
            return NAV_QUIT
        if choice in valid:
            selected_key = valid[choice]
            # Check if user selected the Back option
            if allow_back and selected_key == str(len(options) + 1):
                return NAV_BACK
            return selected_key
        print_error("Invalid choice, try again.")


def ask_positive_int(prompt: str, *, allow_back: bool = False, allow_quit: bool = False) -> int | str:
    while True:
        raw = input(f"{Fore.CYAN}➤ {prompt}: {Style.RESET_ALL}").strip()
        upper = raw.upper()
        if allow_back and upper in BACK_COMMANDS:
            return NAV_BACK
        if allow_quit and upper in QUIT_COMMANDS:
            return NAV_QUIT
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print_error("Please type a positive integer.")


def ask_folder_name(*, allow_quit: bool = False) -> str:
    while True:
        folder = input(f"{Fore.CYAN}➤ Output folder name (relative path): {Style.RESET_ALL}").strip()
        upper = folder.upper()
        if allow_quit and upper in QUIT_COMMANDS:
            return NAV_QUIT
        if not folder:
            print_error("Folder name cannot be empty.")
            continue
        if not FOLDER_NAME_PATTERN.fullmatch(folder):
            print_error("Use only letters, numbers, '.', '_', '-', '/' or '\\\\'.")
            continue
        return folder


def ask_text(prompt: str, *, allow_back: bool = False, allow_quit: bool = False) -> str:
    while True:
        raw = input(prompt).strip()
        upper = raw.upper()
        if allow_back and upper in BACK_COMMANDS:
            return NAV_BACK
        if allow_quit and upper in QUIT_COMMANDS:
            return NAV_QUIT
        return raw


def list_numeric_pages(output_dir: Path) -> List[int]:
    numbers: List[int] = []
    for tex_file in output_dir.glob("*.tex"):
        stem = tex_file.stem
        if stem.isdigit():
            numbers.append(int(stem))
    return sorted(numbers)


def next_page_number(output_dir: Path) -> int:
    pages = list_numeric_pages(output_dir)
    return (pages[-1] + 1) if pages else 1


def total_page_count(output_dir: Path) -> int:
    pages = list_numeric_pages(output_dir)
    return pages[-1] if pages else 0


def write_folder_main_tex(output_dir: Path, total_pages: int) -> None:
    root_main = Path.cwd() / "main" / "main.tex"

    if root_main.exists():
        template = root_main.read_text(encoding="utf-8")
    else:
        template = (
            r"\documentclass[a5paper,11pt]{article}" "\n"
            r"\begin{document}" "\n"
            r"\input{1.tex}" "\n"
            r"\end{document}" "\n"
        )

    content = template
    content = re.sub(
        r"\\newcommand\{\\BookFolder\}\{[^}]*\}",
        lambda _match: r"\newcommand{\BookFolder}{.}",
        content,
    )
    content = re.sub(
        r"\\newcommand\{\\TotalPages\}\{[^}]*\}",
        lambda _match: f"\\newcommand{{\\TotalPages}}{{{total_pages}}}",
        content,
    )
    content = content.replace(
        r"\graphicspath{{extras/}}",
        r"\graphicspath{{../extras/}{extras/}}",
    )
    content = content.replace(
        r"\input{extras/vitorias.tex}",
        r"\IfFileExists{../extras/vitorias.tex}{\input{../extras/vitorias.tex}}{}",
    )

    (output_dir / "main.tex").write_text(content, encoding="utf-8")


def generate_pages(
    *,
    provider: WordProvider,
    page_type: str,
    pages_count: int,
    difficulty: DifficultyConfig,
    output_dir: Path,
    used_tokens: Set[str],
    used_words: List[str],
    used_word_set: Set[str],
    rng: random.Random,
) -> Tuple[int, int]:
    start_page = next_page_number(output_dir)

    print_info(f"Generating {pages_count} page(s) starting from page {start_page}...")

    for offset in range(pages_count):
        page_number = start_page + offset
        print(f"{Fore.CYAN}[{offset + 1}/{pages_count}]{Style.RESET_ALL} Generating page {page_number}...", end=" ", flush=True)

        if page_type in {"A", "B"}:
            length_plan = build_length_plan(14, rng)
            words_a = provider.sample_words(length_plan, used_tokens, used_words)
            register_words(words_a, used_tokens, used_words, used_word_set)

            words_b = provider.sample_words(length_plan, used_tokens, used_words)
            register_words(words_b, used_tokens, used_words, used_word_set)

            grid_a = generate_grid(words_a, rows=14, cols=11, difficulty_bonus=difficulty.bonus, rng=rng)
            grid_b = generate_grid(words_b, rows=14, cols=11, difficulty_bonus=difficulty.bonus, rng=rng)

            page_tex = render_battle_page(
                grid_a=grid_a,
                grid_b=grid_b,
                words_a=words_a,
                words_b=words_b,
                difficulty=difficulty,
                with_details=(page_type == "A"),
            )
        elif page_type == "C":
            length_plan = build_length_plan(20, rng)
            words = provider.sample_words(length_plan, used_tokens, used_words)
            register_words(words, used_tokens, used_words, used_word_set)

            grid = generate_grid(words, rows=28, cols=17, difficulty_bonus=difficulty.bonus, rng=rng)
            page_tex = render_type_c_page(
                grid=grid,
                words=words,
                difficulty=difficulty,
                grid_on_left=True,
            )
        elif page_type == "D":
            length_plan = build_length_plan(27, rng)
            words = provider.sample_words(length_plan, used_tokens, used_words)
            register_words(words, used_tokens, used_words, used_word_set)

            grid = generate_grid(words, rows=20, cols=20, difficulty_bonus=difficulty.bonus, rng=rng)
            page_tex = render_type_d_page(
                grid=grid,
                words=words,
                difficulty=difficulty,
                words_on_top=True,
            )
        elif page_type == "E":
            # Type E: Like Type B but with 16 words each (14+2) and 14 cols (11+3)
            length_plan = build_length_plan(16, rng)
            words_a = provider.sample_words(length_plan, used_tokens, used_words)
            register_words(words_a, used_tokens, used_words, used_word_set)

            words_b = provider.sample_words(length_plan, used_tokens, used_words)
            register_words(words_b, used_tokens, used_words, used_word_set)

            grid_a = generate_grid(words_a, rows=14, cols=14, difficulty_bonus=difficulty.bonus, rng=rng)
            grid_b = generate_grid(words_b, rows=14, cols=14, difficulty_bonus=difficulty.bonus, rng=rng)

            page_tex = render_type_e_page(
                grid_a=grid_a,
                grid_b=grid_b,
                words_a=words_a,
                words_b=words_b,
                difficulty=difficulty,
            )
        else:  # Type F
            # Type F: 30 rows (24+6), 12 cols, 22 words (18+4)
            length_plan = build_length_plan(22, rng)
            words = provider.sample_words(length_plan, used_tokens, used_words)
            register_words(words, used_tokens, used_words, used_word_set)

            grid = generate_grid(words, rows=30, cols=12, difficulty_bonus=difficulty.bonus, rng=rng)
            page_tex = render_type_f_page(
                grid=grid,
                words=words,
                difficulty=difficulty,
            )

        file_path = output_dir / f"{page_number}.tex"
        file_path.write_text(page_tex, encoding="utf-8")
        print(f"{Fore.GREEN}✓{Style.RESET_ALL}")

    total_pages = total_page_count(output_dir)
    write_folder_main_tex(output_dir=output_dir, total_pages=total_pages)
    return start_page, start_page + pages_count - 1


def main() -> None:
    rng = random.Random()

    clear_screen()
    print_header("🧩 Soup of Letters Book Generator 🧩")
    print_info("Tip: You can type 'Q' or 'QUIT' at any time to exit")
    print()

    source_choice = ask_choice(
        "Word source:",
        [
            ("1", "JSON files (palavras/4..8.json)"),
            ("2", "AI generated words"),
        ],
        allow_quit=True,
    )
    if source_choice == NAV_QUIT:
        clear_screen()
        print_success("Goodbye! 👋")
        return

    clear_screen()
    print_header("🧩 Soup of Letters Book Generator 🧩")
    folder_name = ask_folder_name(allow_quit=True)
    if folder_name == NAV_QUIT:
        clear_screen()
        print_success("Goodbye! 👋")
        return

    output_dir = Path(folder_name)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print_success(f"Output folder: {output_dir}")

    current_source = source_choice
    json_provider: JsonWordProvider | None = None
    if current_source == "1":
        print_info("Loading word pools from JSON files...")
        word_pools = load_word_pools(Path.cwd() / "palavras")
        json_provider = JsonWordProvider(word_pools, rng)
        print_success("Word pools loaded successfully!")

    used_tokens: Set[str] = set()
    used_words: List[str] = []
    used_word_set: Set[str] = set()

    while True:
        clear_screen()
        print_header("🧩 Soup of Letters Book Generator 🧩")

        source_label = f"{Fore.GREEN}JSON files{Style.RESET_ALL}" if current_source == "1" else f"{Fore.MAGENTA}AI generated{Style.RESET_ALL}"
        total_pages = total_page_count(output_dir)

        info_lines = [
            f"Output folder: {output_dir}",
            f"Word source: {source_label}",
            f"Total pages: {total_pages}",
        ]
        print_box(info_lines, Fore.CYAN)

        menu_choice = ask_choice(
            "Main Menu",
            [
                ("1", "Generate pages"),
                ("2", "Change word source"),
                ("3", "Quit"),
            ],
            allow_quit=True,
        )

        if menu_choice in {"3", NAV_QUIT}:
            break

        if menu_choice == "2":
            clear_screen()
            print_header("🧩 Soup of Letters Book Generator 🧩")
            source_choice = ask_choice(
                "Word source:",
                [
                    ("1", "JSON files (palavras/4..8.json)"),
                    ("2", "AI generated words"),
                ],
                allow_back=True,
                allow_quit=True,
            )
            if source_choice == NAV_BACK:
                continue
            if source_choice == NAV_QUIT:
                break

            current_source = source_choice
            if current_source == "1":
                print_info("Loading word pools from JSON files...")
                word_pools = load_word_pools(Path.cwd() / "palavras")
                json_provider = JsonWordProvider(word_pools, rng)
                print_success("Word pools loaded successfully!")
                input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
            continue

        # Clear before page type selection
        clear_screen()
        print_header("🧩 Soup of Letters Book Generator 🧩")
        page_type = ask_choice(
            "Page type:",
            [
                ("A", "Battle (2 puzzles + date/winner/time)"),
                ("B", "Battle single-player (2 puzzles, no details)"),
                ("C", "Single puzzle 28x17 with 20 words"),
                ("D", "Square puzzle 20x20 with 27 words (6 columns)"),
                ("E", "Battle 14x14 with 16 words each (no details)"),
                ("F", "Tall format 30x12 with 22 words (side list)"),
            ],
            allow_back=True,
            allow_quit=True,
        )
        if page_type == NAV_BACK:
            continue
        if page_type == NAV_QUIT:
            break

        # Clear before pages count input
        clear_screen()
        print_header("🧩 Soup of Letters Book Generator 🧩")
        pages_count = ask_positive_int(
            "How many pages to generate?",
            allow_back=True,
            allow_quit=True,
        )
        if pages_count == NAV_BACK:
            continue
        if pages_count == NAV_QUIT:
            break

        # Clear before difficulty selection
        clear_screen()
        print_header("🧩 Soup of Letters Book Generator 🧩")
        difficulty_choice = ask_choice(
            "Difficulty:",
            [
                ("1", "Fácil"),
                ("2", "Médio"),
                ("3", "Dificil"),
                ("4", "Impossível"),
            ],
            allow_back=True,
            allow_quit=True,
        )
        if difficulty_choice == NAV_BACK:
            continue
        if difficulty_choice == NAV_QUIT:
            break
        difficulty = DIFFICULTIES[difficulty_choice]

        themes: List[str] = []
        if current_source == "2":
            clear_screen()
            print_header("🧩 Soup of Letters Book Generator 🧩")
            print_info("Enter themes separated by commas (e.g., 'animals, food, sports')")
            print_info("Leave blank for random themes")
            raw_themes = ask_text(
                f"\n{Fore.CYAN}➤ Themes: {Style.RESET_ALL}",
                allow_back=True,
                allow_quit=True,
            )
            if raw_themes == NAV_BACK:
                continue
            if raw_themes == NAV_QUIT:
                break
            if raw_themes:
                themes = [part.strip() for part in raw_themes.split(",") if part.strip()]
                print_success(f"Using themes: {', '.join(themes)}")

        clear_screen()
        print_header("🧩 Soup of Letters Book Generator 🧩")

        try:
            if current_source == "1":
                if json_provider is None:
                    print_info("Loading word pools from JSON files...")
                    word_pools = load_word_pools(Path.cwd() / "palavras")
                    json_provider = JsonWordProvider(word_pools, rng)
                    print_success("Word pools loaded successfully!")
                provider: WordProvider = json_provider
            else:
                print_info("Initializing AI word provider...")
                provider = AIWordProvider(themes=themes, rng=rng)
                print_success("AI provider ready!")

            first_page, last_page = generate_pages(
                provider=provider,
                page_type=page_type,
                pages_count=int(pages_count),
                difficulty=difficulty,
                output_dir=output_dir,
                used_tokens=used_tokens,
                used_words=used_words,
                used_word_set=used_word_set,
                rng=rng,
            )

            print()
            result_lines = [
                f"✓ Generated pages {first_page} to {last_page}",
                f"✓ Saved to: {output_dir}",
                f"✓ Updated: {output_dir / 'main.tex'}",
            ]
            print_box(result_lines, Fore.GREEN)

            input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
        except Exception as error:
            print()
            print_error(f"Error: {error}")
            input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")

    clear_screen()
    print_header("🧩 Soup of Letters Book Generator 🧩")
    print_success("Thank you for using the generator! 👋")
    print()


if __name__ == "__main__":
    main()
