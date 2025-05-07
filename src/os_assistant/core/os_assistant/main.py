from collections.abc import Sequence


def main(argv: Sequence[str] | None) -> int:
    print(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
