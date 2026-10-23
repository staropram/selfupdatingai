from selfupdatingai.ai_interface import AIInterface
from selfupdatingai.html_ui import HTMLUI
import tomllib
from pathlib import Path

def main():
    # book-keeping, load config
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    CONFIG_PATH = PROJECT_ROOT / "config.toml"
    with CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)

    ai_interface = AIInterface(config)
    html_ui = HTMLUI(ai_interface)
    html_ui.launch()

if __name__=="__main__":
    main()