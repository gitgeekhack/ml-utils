class InvalidSplittingValues(Exception):

    def __init__(self, value,
                 message='One or more values are invalid {} \n The allowed values must be greater than 0, less than 1 and sum of the these values must be 1'):
        self.message = message.format([", ".join(f'{k}={v}' for k, v in value.items())][0])
        super().__init__(self.message)

    def __str__(self):
        return self.message


class InsufficientData(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class DirectoryNotFound(Exception):

    def __init__(self, message, dir_path):
        self.message = message
        self.dir_path = dir_path
        super().__init__(self.message)

    def __str__(self):
        return f'{self.dir_path} -> {self.message}'


class FileNotFound(Exception):

    def __init__(self, message, file_path):
        self.message = message
        self.file_path = file_path
        super().__init__(self.message)

    def __str__(self):
        return f'{self.file_path} -> {self.message}'


class InvalidConfiguration(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class UnsupportedObjectType(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class MissingRequiredParameterException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'
