"""Physical geometry contracts for the opt-in face approach experiment."""
import numpy as np
import pytest

from srb.nmpc.surface_approach import face_goal, surface_sample, largest_axis_face, staged_reference
from srb.nmpc.quaternion import quaternion_to_rotation


def test_synchronous_reference_moves_and_turns_together_with_consistent_velocity():
    from srb.nmpc.surface_approach import synchronous_reference
    from scipy.spatial.transform import Rotation
    p0 = np.array([-3., .2, .15]); pg = np.array([-.5, 0., 0.])
    com = np.array([.003, .004, .045])
    def ref(t):
        return synchronous_reference(np.atleast_1d(t), p0, [1.,0.,0.,0.],
                                     pg, [np.sqrt(.5),0.,np.sqrt(.5),0.], com, 40.)
    for t in [5., 20., 35.]:
        x = ref(t)[0]; r = quaternion_to_rotation(x[6:10])
        progress = ((x[:3]-r@com)-p0)[0]/(pg-p0)[0]
        angle = Rotation.from_matrix(r).as_rotvec()[1]
        assert 0 < progress < 1
        assert angle/(np.pi/2) == pytest.approx(progress)
        eps = 1e-4
        np.testing.assert_allclose(x[3:6], (ref(t+eps)[0,:3]-ref(t-eps)[0,:3])/(2*eps), atol=1e-8)
        rm = quaternion_to_rotation(ref(t-eps)[0,6:10])
        rp = quaternion_to_rotation(ref(t+eps)[0,6:10])
        np.testing.assert_allclose(x[10:13], Rotation.from_matrix(rm.T@rp).as_rotvec()/(2*eps), atol=1e-8)
    for t, p in [(-1.,p0),(0.,p0),(40.,pg),(50.,pg)]:
        x = ref(t)[0]
        np.testing.assert_allclose(x[:3]-quaternion_to_rotation(x[6:10])@com, p, atol=1e-12)
        np.testing.assert_allclose(x[[3,4,5,10,11,12]], 0., atol=1e-12)


@pytest.mark.parametrize('duration', [0., -1., np.nan, np.inf])
def test_synchronous_reference_rejects_invalid_duration(duration):
    from srb.nmpc.surface_approach import synchronous_reference
    with pytest.raises(ValueError):
        synchronous_reference([0.], np.zeros(3), [1.,0.,0.,0.], np.ones(3),
                              [1.,0.,0.,0.], np.zeros(3), duration)


def test_face_gap_and_alignment_follow_rotated_translated_target():
    target_q = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    target_p = np.array([2., -3., 1.])
    top = np.array([0., 0., .15])
    front = np.array([-.05, 0., 0.])
    p, q = face_goal(target_p, target_q, front, top, .30)
    rt, ra = quaternion_to_rotation(target_q), quaternion_to_rotation(q)
    n = rt @ [-1., 0., 0.]
    np.testing.assert_allclose(ra @ [0., 0., 1.], -n, atol=1e-12)
    np.testing.assert_allclose(ra @ [0., 1., 0.], rt @ [0., 1., 0.], atol=1e-12)
    np.testing.assert_allclose(p + ra @ top - (target_p + rt @ front), .30*n, atol=1e-12)


def test_surface_metrics_remove_rotated_com_and_include_appendages():
    top, front = np.array([0., 0., .15]), np.array([-.05, 0., 0.])
    p, q = face_goal([0., 0., 0.], [1., 0., 0., 0.], front, top, .30)
    com = [.02, -.01, .04]
    x = np.zeros(13); x[:3] = p + quaternion_to_rotation(q) @ com; x[6:10] = q
    # Antenna extends .07 m ahead of the target face; top appendage extends .05 m.
    sample = surface_sample(x, com, top, front, [[0., 0., .20]], [[-.12, 0., 0.]])
    assert sample['face_gap_m'] == pytest.approx(.30)
    assert sample['lateral_error_m'] == pytest.approx(0., abs=1e-12)
    assert sample['facing_error_deg'] == pytest.approx(0., abs=1e-6)
    assert sample['mesh_plane_gap_m'] == pytest.approx(.18)
    # Crossing the plane must not turn into a positive Euclidean distance.
    x[0] += .25
    assert surface_sample(x, com, top, front, [[0.,0.,.20]], [[-.12,0.,0.]])['mesh_plane_gap_m'] < 0


def test_face_extraction_uses_bus_polygon_not_outer_appendage_extent():
    points = np.array([[-.05,-.05,.15],[.05,-.05,.15],[.05,.05,.15],[-.05,.05,.15],
                       [0.,0.,.20],[.01,0.,.20],[.01,.01,.20],[0.,.01,.20]])
    face = largest_axis_face(points, [[0,1,2,3],[4,5,6,7]], [0.,0.,1.])
    np.testing.assert_allclose(face, [0.,0.,.15])
    with pytest.raises(ValueError, match='face'):
        largest_axis_face(points, [[0,1,2,3]], [-1.,0.,0.])


def test_staged_reference_has_consistent_com_velocity_and_settles_at_face_goal():
    p0 = np.array([-3.,.2,.15]); q0 = [1.,0.,0.,0.]; com = np.array([.003,.004,.045])
    pg, qg = face_goal([0.,0.,0.], q0, [-.05,0.,0.], [0.,0.,.15], .30)
    def ref(t):
        return staged_reference(np.atleast_1d(t), p0, q0, pg, qg, com, 12., 40.)
    np.testing.assert_allclose(ref(0)[0,:3], p0+com)
    for t in [5.,20.,45.]:
        epsilon = 1e-4
        derivative = (ref(t+epsilon)[0,:3] - ref(t-epsilon)[0,:3])/(2*epsilon)
        np.testing.assert_allclose(ref(t)[0,3:6], derivative, atol=1e-8)
    np.testing.assert_allclose(ref([52.,60.])[:,:3], np.tile(pg+quaternion_to_rotation(qg)@com,(2,1)), atol=1e-12)
    np.testing.assert_allclose(ref([0.,12.,52.,60.])[:,[3,4,5,10,11,12]], 0., atol=1e-12)
    np.testing.assert_allclose(ref(5)[0,:3]-quaternion_to_rotation(ref(5)[0,6:10])@com, p0, atol=1e-12)
