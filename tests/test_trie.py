from lie.trie import Trie, Element


def test_add_and_find():
    tr = Trie()
    e1 = Element()
    guid = tr.add(e1)
    e1_ = tr.find(guid)
    assert e1_.id == e1.id


def make_trie():
    tr = Trie()
    tr.root.attrib = 0
    elems = [Element() for _ in range(12)]
    for i, e in enumerate(elems):
        e.attrib = i + 1
    for e in elems[0:3]:
        tr.add(e)
    for e in elems[3:6]:
        tr.add(e, elems[0])
    for e in elems[6:9]:
        tr.add(e, elems[1])
    for e in elems[9:12]:
        tr.add(e, elems[2])

    return tr


def test_make_trie():
    tr = make_trie()
    assert [e.attrib for e in tr.flatten()] == [4, 5, 6, 1, 7, 8, 9, 2, 10, 11, 12, 3, 0]


def test_delete_trie():
    tr = make_trie()
    tr.delete(tr.children(tr.root)[0])
    assert [e.attrib for e in tr.flatten()] == [7, 8, 9, 2, 10, 11, 12, 3, 0]

    tr = make_trie()
    tr.delete(tr.children(tr.root)[1])
    assert [e.attrib for e in tr.flatten()] == [4, 5, 6, 1, 10, 11, 12, 3, 0]

    tr = make_trie()
    tr.delete(tr.children(tr.root)[2])
    assert [e.attrib for e in tr.flatten()] == [4, 5, 6, 1, 7, 8, 9, 2, 0]

    tr = make_trie()
    tr.delete(tr.children(tr.children(tr.root)[0])[0])
    assert [e.attrib for e in tr.flatten()] == [5, 6, 1, 7, 8, 9, 2, 10, 11, 12, 3, 0]


def test_path():
    tr = make_trie()
    guid = tr.children(tr.children(tr.root)[0])[0].id
    p = tr.path(guid)
    assert [e.attrib for e in p] == [0, 1, 4]