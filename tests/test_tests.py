import pytest


@pytest.mark.parametrize('word', ['Hello World!', 'DeEZ NuTs'])
def test_print(log, capsys, word):

    log.debug(word)
    log.info(word)
    log.warning(word)
    log.error(word)
    log.critical(word)

    print(word)
    captured = capsys.readouterr()
    assert captured.out.strip('\n') == word
