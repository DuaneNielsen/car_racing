import uuid


def flatten(element):
    l = []
    for guid, e in element.children.items():
        l += flatten(e)
    l.append(element)
    return l


def path(element):
    p = [element]
    while element.parent is not None:
        p += [element.parent]
        element = element.parent
    return reversed(p)


class Element:
    def __init__(self):
        self.parent = None
        self.id = uuid.uuid4()
        self.children = {}

    def path(self):
        return path(self)

    def flatten(self):
        return flatten(self)


class Trie:
    def __init__(self):
        self.root = Element()
        self.elements = {str(self.root.id): self.root}

    def find(self, guid):
        if str(guid) in self.elements:
            return self.elements[str(guid)]
        else:
            return None

    def add(self, child, parent=None):
        if parent is None:
            parent = self.root
        parent.children[str(child.id)] = child
        self.elements[str(child.id)] = child
        child.parent = parent
        return child.id

    def children(self, e):
        return [e for guid, e in e.children.items()]

    def flatten(self, guid=None):
        if guid is not None:
            if guid in self.elements:
                return flatten(self.find(guid))
            else:
                return None
        else:
            return flatten(self.root)

    def delete(self, element):
        for e in element.flatten():
            del self.elements[str(e.id)]
        del element.parent.children[str(element.id)]

    def path(self, guid=None):
        element = self.find(guid)
        if element is not None:
            return path(element)
        else:
            return None
