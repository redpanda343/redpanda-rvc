class I18nAuto:
    def __init__(self, language=None):
        self.language = "en_US"

    def __call__(self, key):
        return key
