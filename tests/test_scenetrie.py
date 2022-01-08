import scenetrie as st
import lie.SE2 as SE2
import jax.numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon


def test_le_rotation():
    scene = st.Scene()
    fsa_guid = scene.add(st.Frame(transform=SE2.from_xytheta(theta=0.)))
    fsb_guid = scene.add(st.Frame(transform=SE2.from_xytheta(theta=np.pi/2.)))
    fsc_guid = scene.add(st.Frame(transform=SE2.from_xytheta(theta=np.pi)))
    fsd_guid = scene.add(st.Frame(transform=SE2.from_xytheta(theta=-np.pi/2)))
    pa = np.array([[1.], [1.]])
    pb = np.array([[-1.], [1.]])
    pc = np.array([[-1.], [-1.]])
    pd = np.array([[1.], [-1.]])

    pa_guid = scene.add(st.Points(p=pa), parent=scene.find(fsa_guid))

    tab = st.change_frame(from_frame=scene[fsa_guid], to_frame=scene[fsb_guid])
    assert np.allclose(pb, scene[pa_guid].apply_SE2_matrix(tab), atol=1e-5)

    tac = st.change_frame(from_frame=scene[fsa_guid], to_frame=scene[fsc_guid])
    assert np.allclose(pc, scene[pa_guid].apply_SE2_matrix(tac), atol=1e-5)

    tad = st.change_frame(from_frame=scene[fsa_guid], to_frame=scene[fsd_guid])
    assert np.allclose(pd, scene[pa_guid].apply_SE2_matrix(tad), atol=1e-5)


def test_cam_lerp():
    fig, ax = plt.subplots()
    scene = st.Scene()
    rect_guid = scene.add(st.Rectangle(h=10., w=10.))
    camera_frame_guid = scene.add(st.Rectangle(h=20, w=20))

    def draw_poly(ax, poly, camera):
        poly_frame_to_cam_frame = st.change_frame(from_frame=camera, to_frame=poly)
        verts = poly.apply_SE2_matrix(poly_frame_to_cam_frame)
        ax.add_patch(Polygon(verts.T, color='blue', fill=False))

    for x in np.linspace(0, 10., 20):
        scene[camera_frame_guid].transform = SE2.from_xytheta(x=x)
        ax.clear()
        draw_poly(ax, scene[rect_guid], scene[camera_frame_guid])
        draw_poly(ax, scene[camera_frame_guid], scene[camera_frame_guid])
        ax.set_xlim(-1, 50.)
        ax.set_ylim(-1, 50.)
        plt.pause(1.0)

    plt.show()

def test_scene():
    fig, ax = plt.subplots()
    scene = st.Scene()
    rect_guid = scene.add(st.Rectangle(h=10., w=10.))
    circle_guid = scene.add(st.Circle(h=10., w=10.))
    camera_frame_guid = scene.add(st.Rectangle(h=20, w=20))

    def draw_poly(ax, poly, camera):
        poly_frame_to_cam_frame = st.change_frame(from_frame=camera, to_frame=poly)
        verts = poly.apply_SE2_matrix(poly_frame_to_cam_frame)
        ax.add_patch(Polygon(verts.T, color='blue', fill=False))

    for x in np.linspace(0, 10., 20):
        scene[camera_frame_guid].transform = SE2.from_xytheta(x=x)
        ax.clear()
        draw_poly(ax, scene[rect_guid], scene[camera_frame_guid])
        draw_poly(ax, scene[camera_frame_guid], scene[camera_frame_guid])
        ax.set_xlim(-1, 50.)
        ax.set_ylim(-1, 50.)
        plt.pause(1.0)

    plt.show()


