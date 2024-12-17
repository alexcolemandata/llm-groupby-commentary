from . import happy


def main() -> None:
    print(happy.read_data_folder(happy.SOURCE_DIR).collect())

    return None
