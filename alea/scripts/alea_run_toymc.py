from alea.runner import Runner
from alea.submitter import Submitter


def main():
    kwargs = Submitter.runner_kwargs_from_script()
    runner = Runner(**kwargs)
    runner.run()


if __name__ == "__main__":
    main()
