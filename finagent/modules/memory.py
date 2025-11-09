# finagent/modules/memory.py
class MemoryFacade:
    def __init__(self, store):
        self.store = store
    # We already write via each module; this class is here if you want unified APIs later.
