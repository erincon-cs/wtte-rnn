def boolean(arg: str):
    """
    Converts an argument string and converts it to a str
    :param arg: (str) argument string
    :return: (bool) True or False
    """

    arg = arg.lower()

    if arg in ("1", "true", "yes", "y"):
        return True
    else:
        return False
