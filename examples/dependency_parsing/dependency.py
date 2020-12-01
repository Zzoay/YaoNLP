

class Dependency():
    def __init__(self, id, word, tag, head, rel):
        self.id = id
        self.word = word
        self.tag = tag
        self.head = head
        self.rel = rel

    def __str__(self):
        # example:  1	上海	_	NR	NR	_	2	nn	_	_
        values = [str(self.id), self.word, "_", self.tag, "_", "_", str(self.head), self.rel, "_", "_"]
        return '\t'.join(values)

    def __repr__(self):
        return f"({self.word} {self.tag})"