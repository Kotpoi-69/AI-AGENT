from ai_orchestrator import run_ai_cycle
import time


def main():

    while True:

        result = run_ai_cycle()

        print("Cycle Done:", result)


if __name__ == "__main__":
    main()
