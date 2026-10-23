def test_max_file_size():
    from services.reader import MAX_FILE_SIZE
    assert MAX_FILE_SIZE == 10 * 1024 * 1024


def test_import_main():
    import main
    assert main.app is not None
