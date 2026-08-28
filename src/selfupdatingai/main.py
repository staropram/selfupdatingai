from selfupdatingai.ai_interface import AIInterface
from selfupdatingai.html_ui import HTMLUI

def main():
    ai_interface = AIInterface()
    html_ui = HTMLUI(ai_interface)
    html_ui.launch()

if __name__=="__main__":
    main()