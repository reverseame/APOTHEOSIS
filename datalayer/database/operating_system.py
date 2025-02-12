
class OS:
    def __init__(self, id, name, version):
        self.id = id
        self.name = name
        self.version = version

    def __str__(self):
        return f"Name: {self.name}, Version: {self.version}"
    
    def as_dict(self):
        return {f"os_{key}": value for key, value in self.__dict__.items()}

    def __eq__(self, other):
        if isinstance(other, OS):
            return self.id == other.id and self.name == other.name and self.version == other.version
        return False

    def __hash__(self):
        return hash((self.id, self.name, self.version))
