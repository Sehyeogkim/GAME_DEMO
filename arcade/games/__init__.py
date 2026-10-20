"""Game registry: name -> (module path, class name, port)."""
import importlib

GAMES = {
    "domino_run": ("arcade.games.domino_run.game", "DominoRun", 8801),
    "claw_prize": ("arcade.games.claw_prize.game", "ClawPrize", 8802),
    "color_sort": ("arcade.games.color_sort.game", "ColorSort", 8803),
    "tower_stack": ("arcade.games.tower_stack.game", "TowerStack", 8804),
    "bowling_setup": ("arcade.games.bowling_setup.game", "BowlingSetup", 8805),
    "shape_sorter": ("arcade.games.shape_sorter.game", "ShapeSorter", 8806),
    "drawer_heist": ("arcade.games.drawer_heist.game", "DrawerHeist", 8807),
    "push_golf": ("arcade.games.push_golf.game", "PushGolf", 8808),
}


def load_game(name):
    module_path, class_name, _ = GAMES[name]
    return getattr(importlib.import_module(module_path), class_name)()


def available():
    """Names whose module imports cleanly."""
    found = []
    for name in GAMES:
        try:
            load_game(name)
            found.append(name)
        except ModuleNotFoundError:
            pass
    return found
