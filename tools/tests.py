from logger import create_logger


class Tester():
    def __init__(self, debug):
        self.debug = debug
        self.logger = create_logger(name=__name__, level='DEBUG')

    def test_list(self, l, test):
        if not self.debug:
            return
        else:
            logger = self.logger
            logger.debug('Printing elements...')
            for e in l:
                logger.debug(e)

            if not test(l): logger.warning("Test failed.")
            else: logger.info('Test success.')
    