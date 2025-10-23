import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip().splitlines(True)}


def first_line(cell):
    src = "".join(cell.get("source", []))
    return (src.strip().splitlines() or [""])[0]


def insert_after(cells, anchor_idx, new_cells):
    for off, nc in enumerate(new_cells, start=1):
        cells.insert(anchor_idx + off, nc)


def indices_with_first_line(cells, line):
    return [i for i, c in enumerate(cells) if first_line(c) == line]


def patch_notebook(nb_path: Path):
    nb = json.loads(nb_path.read_text())
    cells = nb.get("cells", [])

    # 0b. Imports from ../../src
    import_md = md("""
    ## 0b. Импорты семинара (из src/)
    """)
    import_block = """# Импортируем модули семинара из ../../src\n""" \
        "    import sys, os\n" \
        "    sys.path.append(os.path.abspath(os.path.join('..','..','src')))\n" \
        "    from rlcourse.sem01.envs import make_cartpole, make_frozenlake, SEED\n" \
        "    from rlcourse.sem01.algorithms import run_random, eval_value_mc, q_learning, td0_value_learning, moving_avg, smooth, eval_greedy\n" \
        "    from rlcourse.sem01.discretization import discretize, make_edges, BIN_COUNTS, bounds\n" \
        "    from rlcourse.sem01.policies import epsilon_greedy, softmax_action, random_policy_from_env\n" \
        "    from rlcourse.sem01.video import record_episodes\n"
    import_code = code(import_block)

    # Insert after first title markdown if not present
    idxs = indices_with_first_line(cells, '## 0b. Импорты семинара (из src/)')
    if not idxs:
        insert_pos = 0
        if cells and cells[0].get('cell_type') == 'markdown':
            insert_pos = 0
        insert_after(cells, insert_pos, [import_md, import_code])
        idxs = indices_with_first_line(cells, '## 0b. Импорты семинара (из src/)')

    # Ensure the following code cell contains the latest import block
    for idx in idxs:
        if idx + 1 < len(cells) and first_line(cells[idx + 1]).startswith('# Импортируем модули семинара'):
            cells[idx + 1]['source'] = import_block.splitlines(True)

    # Update video demo cell to use module import
    for i, c in enumerate(cells):
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c.get('source', []))
        if 'Запись видео и HTML-вставка' in src and 'record_video' in src:
            new_src = src.replace('from record_video import record_episodes', 'from rlcourse.sem01.video import record_episodes')
            cells[i]['source'] = new_src.splitlines(True)

    nb['cells'] = cells
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2))
    return True


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / 'notebooks' / 'sem01_mdp_bellman.ipynb'
    ok = patch_notebook(path)
    print('Patched:', ok, '->', path)
