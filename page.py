class Page:
    def __init__(self, id, num_page, preprocess_method, hashTLSH, hashSSDEEP,
                  hashSDHASH, module_id):
        self.id = id
        self.num_page = num_page
        self.preprocess_method = preprocess_method
        self.hashTLSH = hashTLSH
        self.hashSSDEEP = hashSSDEEP
        self.hashSDHASH = hashSDHASH
        self.module_id = module_id

