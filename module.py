class Module:
    def __init__(self, file_version, original_filename,
                  internal_filename, product_name, company_name,
                  legal_copyright, classification, size, base_address, os_id):
        
        self.file_version = file_version
        self.original_filename = original_filename
        self.internal_filename = internal_filename
        self.product_name = product_name
        self.company_name = company_name
        self.legal_copyright = legal_copyright
        self.classification = classification
        self.size = size
        self.base_address = base_address
        self.pages = []
        
