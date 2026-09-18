"""Pure-numpy tests for the generic preprocess/postprocess (no torch/ORT needed)."""

import numpy as np
import pytest

from onnx_infer.meta import InputSpec, ModelMeta, Normalize
from onnx_infer.preprocess import Transform, _resize_chw, preprocess
from onnx_infer.postprocess import to_friendy


def _meta(**input_kwargs):
    return ModelMeta(
        arch="retinanet",
        num_classes=2,
        class_map={0: "a", 1: "b"},
        score_threshold=0.05,
        input=InputSpec(**input_kwargs),
        normalize=None,
    )


def test_resize_none_is_identity():
    img = np.random.rand(3, 40, 50).astype(np.float32)
    batched, tf = preprocess(img, _meta(resize_mode="none"))
    assert batched.shape == (1, 3, 40, 50)
    np.testing.assert_array_equal(batched[0], img)
    assert (tf.scale_x, tf.scale_y, tf.pad_x, tf.pad_y) == (1.0, 1.0, 0, 0)
    assert (tf.orig_w, tf.orig_h) == (50, 40)


def test_pad_to_multiple_bottom_right():
    img = np.ones((3, 30, 50), dtype=np.float32)
    batched, tf = preprocess(img, _meta(resize_mode="none", multiple=32, pad_value=0.0))
    assert batched.shape == (1, 3, 32, 64)  # ceil(30->32), ceil(50->64)
    assert np.all(batched[0, :, :30, :50] == 1.0)      # original preserved, top-left
    assert np.all(batched[0, :, 30:, :] == 0.0)         # bottom pad
    assert np.all(batched[0, :, :, 50:] == 0.0)         # right pad
    assert (tf.pad_x, tf.pad_y) == (0, 0)               # bottom-right pad => no origin shift


def test_normalize_applied():
    img = np.full((3, 8, 8), 0.5, dtype=np.float32)
    meta = ModelMeta(
        arch="rtdetr", num_classes=1, class_map={0: "a"}, score_threshold=0.5,
        input=InputSpec(resize_mode="none"),
        normalize=Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
    )
    batched, _ = preprocess(img, meta)
    np.testing.assert_allclose(batched[0], 0.0, atol=1e-6)  # (0.5-0.5)/0.25 == 0


def test_byte_scale():
    img = np.full((3, 4, 4), 1.0, dtype=np.float32)
    batched, _ = preprocess(img, _meta(resize_mode="none", input_scale="byte"))
    np.testing.assert_allclose(batched[0], 255.0, atol=1e-4)


def test_resize_constant_preserved():
    img = np.full((3, 10, 12), 0.7, dtype=np.float32)
    out = _resize_chw(img, 20, 8)
    assert out.shape == (3, 20, 8)
    np.testing.assert_allclose(out, 0.7, atol=1e-6)  # bilinear of a constant is constant


def test_square_transform_scales():
    img = np.random.rand(3, 100, 200).astype(np.float32)
    batched, tf = preprocess(img, _meta(resize_mode="square", size=50))
    assert batched.shape == (1, 3, 50, 50)
    assert tf.scale_x == 50 / 200
    assert tf.scale_y == 50 / 100


def test_to_friendy_maps_back_identity_transform():
    # xyxy [10,20,30,60] in a 100(w) x 200(h) image; scale=1, pad=0.
    boxes = np.array([[10, 20, 30, 60]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    labels = np.array([1], dtype=np.int64)
    tf = Transform(scale_x=1.0, scale_y=1.0, pad_x=0, pad_y=0, orig_w=100, orig_h=200)
    out = to_friendy(boxes, scores, labels, tf, score_threshold=0.05)
    # cx=20,cy=40,w=20,h=40 -> normalized by (100,200)
    np.testing.assert_allclose(out[0, :4], [0.2, 0.2, 0.2, 0.2], atol=1e-6)
    assert out[0, 4] == pytest.approx(0.9)
    assert out[0, 5] == 1.0


def test_to_friendy_inverts_scale_and_pad():
    # A box in a resized+padded frame maps back through the recorded transform.
    tf = Transform(scale_x=0.5, scale_y=0.5, pad_x=4, pad_y=6, orig_w=100, orig_h=100)
    # original box xyxy [20,20,40,40] -> input px: *0.5 then +pad = [14,16,24,26]
    boxes = np.array([[14, 16, 24, 26]], dtype=np.float32)
    out = to_friendy(boxes, np.array([0.8], np.float32), np.array([0], np.int64), tf, 0.05)
    # back to original: cx=30,cy=30,w=20,h=20 over 100 -> 0.3,0.3,0.2,0.2
    np.testing.assert_allclose(out[0, :4], [0.3, 0.3, 0.2, 0.2], atol=1e-5)


def test_to_friendy_threshold_and_empty():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=np.float32)
    scores = np.array([0.9, 0.1], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    tf = Transform(1.0, 1.0, 0, 0, 100, 100)
    out = to_friendy(boxes, scores, labels, tf, score_threshold=0.5)
    assert out.shape == (1, 6)  # only the 0.9 box survives
    empty = to_friendy(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                       np.zeros((0,), np.int64), tf, 0.5)
    assert empty.shape == (0, 6)


# --------------------------------------------------------------- pose keypoints

def _rtmo_outputs(dets, keypoints):
    return [np.asarray(dets, dtype=np.float32), np.asarray(keypoints, dtype=np.float32)]


def test_rtmo_hands_out_the_joints_it_spends_on_the_label():
    """The posture label and the raw joints come off the same graph outputs.

    The handler's whole reason for existing is that Contract A has nowhere to
    put a keypoint array, so the joints get turned into a class. They are still
    the most informative thing the graph produced, and ``adapt_keypoints`` is
    how a caller that can draw them gets them — unchanged, and row-matched to
    the boxes ``adapt_outputs`` returns from the same list.
    """
    from onnx_infer.arch import get_handler

    handler = get_handler("rtmo")
    joints = np.zeros((2, 17, 3), dtype=np.float32)
    joints[:, :, 2] = 0.9
    outputs = _rtmo_outputs([[10, 20, 30, 120, 0.9], [40, 50, 60, 150, 0.8]], joints)

    boxes, scores, labels = handler.adapt_outputs(outputs)
    carried = handler.adapt_keypoints(outputs)

    assert carried.shape == (boxes.shape[0], 17, 3)
    np.testing.assert_array_equal(carried, joints)
    assert scores.shape == labels.shape == (2,)


def test_a_box_arch_has_no_joints_to_hand_out():
    """``None``, not an empty array: "this graph has no keypoints" and "this pose
    graph found nobody" are different answers and callers branch on which."""
    from onnx_infer.arch import get_handler

    outputs = [np.zeros((1, 4), np.float32), np.zeros((1,), np.float32),
               np.zeros((1,), np.int64)]
    assert get_handler("retinanet").adapt_keypoints(outputs) is None


def test_keypoints_land_in_the_same_frame_as_their_boxes():
    """Joints normalize through the same inverse transform the boxes do.

    Built as the exact letterbox case the rtmo bundle runs: a 100x50 image
    scaled by 2 into a padded canvas. A joint at the middle of the subject must
    come back at the middle of the *original* image, or a skeleton would be
    drawn offset from the person it belongs to by the padding.
    """
    from onnx_infer.postprocess import keypoints_to_normalized

    transform = Transform(scale_x=2.0, scale_y=2.0, pad_x=10, pad_y=30,
                          orig_w=100, orig_h=50)
    joints = np.array([[[110.0, 80.0, 0.9], [210.0, 130.0, 0.4]]], dtype=np.float32)

    out = keypoints_to_normalized(joints, np.array([0.9]), transform, 0.5)

    assert out.shape == (1, 2, 3)
    # (110 - 10) / 2 = 50 px of 100 wide; (80 - 30) / 2 = 25 px of 50 tall.
    np.testing.assert_allclose(out[0, 0, :2], [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(out[0, 1, :2], [1.0, 1.0], atol=1e-6)
    # Joint confidences are carried, not recomputed.
    np.testing.assert_allclose(out[0, :, 2], [0.9, 0.4], atol=1e-6)


def test_dropped_detections_drop_their_joints_too():
    """One threshold mask for both, so row i is the same person in each.

    Off-by-one here is the failure that matters: the skeletons would still be
    drawn, just on the wrong people.
    """
    from onnx_infer.postprocess import keypoints_to_normalized

    transform = Transform(scale_x=1.0, scale_y=1.0, pad_x=0, pad_y=0,
                          orig_w=10, orig_h=10)
    scores = np.array([0.9, 0.1, 0.7], dtype=np.float32)
    joints = np.zeros((3, 17, 3), dtype=np.float32)
    joints[:, 0, 0] = [1.0, 2.0, 3.0]  # a per-person marker in the first joint

    boxes = np.tile(np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32), (3, 1))
    labels = np.zeros(3, dtype=np.int64)
    friendy = to_friendy(boxes, scores, labels, transform, 0.5)
    out = keypoints_to_normalized(joints, scores, transform, 0.5)

    assert out.shape[0] == friendy.shape[0] == 2
    np.testing.assert_allclose(out[:, 0, 0] * transform.orig_w, [1.0, 3.0], atol=1e-6)
