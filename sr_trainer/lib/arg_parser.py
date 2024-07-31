import argparse

class ArgParser:
    """
    command-line argument parser

    Attributes:
        args: command-line arguments
    """
    def __init__(self):
        """
        Parse command-line arguments
        """
        # parse command-line arguments
        parser = argparse.ArgumentParser(
            add_help=True)
        parser.add_argument('yml_fname',
            help='filename of input parameters (.yml)')
        parser.add_argument('mode', choices=['train', 'test'],
            help='execution mode: train or test')
        self.args = parser.parse_args()

    def get_fname(self):
        """
        Get yml filename

        Returns:
            yml filename
        """
        return self.args.yml_fname

    def get_mode(self):
        """
        Get execution mode (train or test)

        Returns:
            execution mode
        """
        return self.args.mode
