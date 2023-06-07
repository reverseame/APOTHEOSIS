class OperatingSystem:
    def __init__(self, id, name, version):
        self.id = id
        self.version = version
        self.name = name
        self.modules = []

    def __str__(self):
        return f"Name: {self.name}, Version: {self.version}"
