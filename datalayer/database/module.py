
class Module:
    def __init__(self, os, id, file_version, original_filename,
        internal_filename, product_name,company_name, legal_copyright,
        classification, size, base_address
        ):

        self.id = id
        self.file_version = file_version
        self.original_filename = original_filename
        self.internal_filename = internal_filename
        self.product_name = product_name
        self.company_name = company_name
        self.legal_copyright = legal_copyright
        self.classification = classification
        self.size = size
        self.base_address = base_address
        self.os = os

    def __eq__(self, module):
        return self.id == module.id
    
    def __hash__(self):
        return hash(self.id)
        
    def as_dict(self):
        return {
            "id": self.id,
            "file_version": self.file_version,
            "original_filename": self.original_filename,
            "internal_filename": self.internal_filename,
            "product_name": self.product_name,
            "company_name": self.company_name,
            "legal_copyright": self.legal_copyright,
            "classification": self.classification,
            "size": self.size,
            "base_address": self.base_address,
            "os": self.os.as_dict() if self.os else None
        }
