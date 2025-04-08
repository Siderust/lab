# Códigos ANSI para colores
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

def colorize(value):
    color = GREEN if value >= 0 else RED
    return f"{color}{value:.10f}{RESET}"
