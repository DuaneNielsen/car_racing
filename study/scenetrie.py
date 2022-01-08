import trie
import jax.numpy as np
import lie.SE2 as SE2
import geometry as geo


class Frame(trie.Element):
    def __init__(self, initial=None, tangent=None, transform=None):
        super().__init__()
        self.initial = initial if initial is not None else np.zeros(3)
        self.tangent = tangent if initial is not None else np.zeros(3)
        self.transform = transform if initial is not None else np.zeros(3)


class Points(Frame):
    def __init__(self, p=None, initial=None, tangent=None, transform=None):
        super().__init__(initial, tangent, transform)
        self.p = p

    def as_homo(self):
        return np.concatenate([self.p, np.ones((1, self.p.shape[1]))])

    def apply_SE2_matrix(self, t):
        return np.matmul(t, self.as_homo())[0:2]


class Polygon(Frame):
    def __init__(self, verts=None, initial=None, tangent=None, transform=None):
        """
        Polygon
        :param verts: (2, 1), counter clockwise winding
        """
        super().__init__(initial, tangent, transform)
        self.verts = verts

    def as_homo(self):
        return np.concatenate([self.verts, np.ones((1, self.verts.shape[1]))])

    def apply_SE2_matrix(self, t):
        return np.matmul(t, self.as_homo())[0:2]


class Rectangle(Polygon):
    def __init__(self, h, w, initial=None, tangent=None, transform=None):
        super().__init__(initial, tangent, transform)
        self.verts = np.array([[0., w, w, 0], [0., 0., h, h]])


class Circle(Polygon):
    def __init__(self, r, initial=None, tangent=None, transform=None):
        super().__init__(initial, tangent, transform)
        self.r = r


class Scene(trie.Trie):
    def __init__(self):
        super().__init__()
        self.root = Frame(
            initial=SE2.from_xytheta(),
            tangent=SE2.from_xytheta(),
            transform=SE2.from_xytheta()
        )


def to_space(frame):
    f = np.eye(3)
    for e in reversed(trie.path(frame)):
        f = np.matmul(SE2.inv_transform_matrix(e.transform), f)
    return f


def change_frame(from_frame, to_frame):
    f = to_space(from_frame)
    for e in trie.path(to_frame):
        f = np.matmul(SE2.transform_matrix(e.transform), f)
    return f
