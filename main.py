from ai_orchestrator import run_ai_cycle
import time


def main():

    while True:

        result = run_ai_cycle()

        print("Cycle Done:", result)
        time.sleep(5)


if __name__ == "__main__":
    main()
