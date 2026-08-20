"""The pure-python half of the FP16/BF16 TensorRT path.

Everything here runs without a GPU, TensorRT or ModelOpt: the precision census
that decides whether a converted graph is *actually* low precision, the profile
resolution that decides what shapes an engine accepts, and the engine-inspection
census that answers "did BF16 actually happen". The parts that need real
hardware (a build, a parity gate) live in ``trt_export/trained_fp16_gate.py`` and
are run by hand in the trainer container.
"""

import json
import unittest
from pathlib import Path

from friendy_chachkalica.ml.trt_export import arch as arch_policy
from friendy_chachkalica.ml.trt_export import build_trt, engine_inspect, modelopt_cast


class _Tensor:
    def __init__(self, name, data_type):
        self.name = name
        self.data_type = data_type


class _ValueInfo:
    class _Type:
        class _TensorType:
            def __init__(self, elem_type):
                self.elem_type = elem_type

        def __init__(self, elem_type):
            self.tensor_type = self._TensorType(elem_type)

    def __init__(self, name, elem_type):
        self.name = name
        self.type = self._Type(elem_type)


class _Node:
    def __init__(self, op_type, inputs, outputs):
        self.op_type = op_type
        self.input = inputs
        self.output = outputs


class _Graph:
    def __init__(self, nodes, initializer=(), value_info=(), inputs=(), outputs=()):
        self.node = list(nodes)
        self.initializer = list(initializer)
        self.value_info = list(value_info)
        self.input = list(inputs)
        self.output = list(outputs)


class _Model:
    def __init__(self, graph):
        self.graph = graph


FLOAT, FLOAT16, BFLOAT16 = 1, 10, 16


class PrecisionCensusTests(unittest.TestCase):
    """``_precision_census`` is what stops a "bf16" engine that is really fp32."""

    def _model(self):
        return _Model(_Graph(
            nodes=[
                _Node("Cast", ["x"], ["x_bf"]),          # plumbing, not compute
                _Node("Conv", ["x_bf", "w"], ["h"]),     # bf16 (bf16 initializer)
                _Node("Softmax", ["h32"], ["p"]),        # kept fp32
                _Node("Cast", ["p"], ["p_out"]),
            ],
            initializer=[_Tensor("w", BFLOAT16)],
            value_info=[_ValueInfo("x_bf", BFLOAT16), _ValueInfo("h", BFLOAT16),
                        _ValueInfo("h32", FLOAT), _ValueInfo("p", FLOAT)],
        ))

    def test_counts_compute_nodes_and_ignores_cast_plumbing(self):
        total, low, fp32_ops, unknown = modelopt_cast._precision_census(self._model(), BFLOAT16)
        self.assertEqual(total, 2)          # Conv + Softmax; the two Casts don't count
        self.assertEqual(low, 1)            # only Conv
        self.assertEqual(fp32_ops, {"Softmax": 1})
        self.assertEqual(unknown, 0)

    def test_integer_plumbing_is_unclassified_not_counted_as_fp32(self):
        """A graph is full of int64 shape/index nodes with no float dtype to judge.
        Folding them into the fp32 side would make every conversion look worse than
        it is; folding them into the denominator would make the fraction meaningless."""
        model = _Model(_Graph(
            nodes=[_Node("Shape", ["x"], ["s"]), _Node("Conv", ["x_bf", "w"], ["h"])],
            initializer=[_Tensor("w", BFLOAT16)],
            value_info=[_ValueInfo("x_bf", BFLOAT16), _ValueInfo("h", BFLOAT16)],
        ))
        total, low, fp32_ops, unknown = modelopt_cast._precision_census(model, BFLOAT16)
        self.assertEqual((total, low, fp32_ops, unknown), (1, 1, {}, 1))

    def test_an_fp16_census_of_a_bf16_graph_finds_nothing(self):
        """Asking the wrong dtype must report zero, not "close enough" — this is the
        check that makes ``cast_onnx_bytes_with_autocast`` refuse a mislabelled graph.

        The bf16 Conv is *unclassified* here rather than fp32: it is neither the
        requested low precision nor float32, so only the genuinely-fp32 Softmax is
        classified."""
        total, low, _, unknown = modelopt_cast._precision_census(self._model(), FLOAT16)
        self.assertEqual((total, low, unknown), (1, 0, 1))


class RequireModelOptTests(unittest.TestCase):
    def test_missing_modelopt_raises_with_the_install_command(self):
        original = modelopt_cast.modelopt_version
        modelopt_cast.modelopt_version = lambda: None
        try:
            with self.assertRaises(modelopt_cast.ModelOptUnavailable) as ctx:
                modelopt_cast.require_modelopt()
        finally:
            modelopt_cast.modelopt_version = original
        self.assertIn("nvidia-modelopt", str(ctx.exception))

    def test_only_fp16_and_bf16_are_autocast_precisions(self):
        with self.assertRaises(ValueError):
            modelopt_cast.cast_onnx_bytes_with_autocast(b"", precision="int8")


class EnginePrecisionCensusTests(unittest.TestCase):
    """``engine_inspect.precision_census`` — the "did BF16 actually happen" reader."""

    INFO = {"Layers": [
        {"Name": "cast_in", "LayerType": "kgen",
         "Inputs": [{"Name": "pixel_values", "Datatype": "Float"}],
         "Outputs": [{"Name": "x_bf", "Datatype": "BFloat16"}]},
        {"Name": "backbone", "LayerType": "CaskConvolution",
         "Inputs": [{"Name": "x_bf", "Datatype": "BFloat16"}],
         "Outputs": [{"Name": "feat", "Datatype": "BFloat16"}]},
        {"Name": "topk", "LayerType": "TopK",
         "Inputs": [{"Name": "feat", "Datatype": "BFloat16"}],
         "Outputs": [{"Name": "scores", "Datatype": "Float"}]},
    ]}

    def test_tensor_census_counts_each_tensor_once(self):
        census = engine_inspect.precision_census(self.INFO)
        # x_bf and feat are each produced once and consumed once.
        self.assertEqual(census["tensors"], {"BFloat16": 2, "Float": 2})
        self.assertEqual(census["num_layers"], 3)

    def test_layer_precision_falls_back_to_the_first_output_dtype(self):
        census = engine_inspect.precision_census(self.INFO)
        self.assertEqual(census["layers"], {"BFloat16": 2, "Float": 1})

    def test_an_explicit_precision_field_wins_over_the_output_dtype(self):
        info = {"Layers": [dict(self.INFO["Layers"][2], Precision="Half")]}
        self.assertEqual(engine_inspect.precision_census(info)["layers"], {"Half": 1})

    def test_empty_engine_information_does_not_crash(self):
        self.assertEqual(engine_inspect.precision_census({})["num_layers"], 0)


class ResolveProfileTests(unittest.TestCase):
    """``build_trt.resolve_profile`` — where an engine's accepted shapes come from.

    Order matters: the meta sidecar is the model's own contract (Contract B) and
    beats a shape guessed off the graph, while explicit flags beat everything.
    A graph with neither is refused rather than guessed at — the profile decides
    what the engine accepts, and for some archs whether a low-precision build
    finds tactics at all.
    """

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.onnx = Path(self.tmp.name) / "model.onnx"
        self._write_onnx(("B", 3, "H", "W"))

    def _write_onnx(self, shape):
        """A one-node graph whose input carries ``shape`` — enough for shape lookup."""
        import onnx
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            [helper.make_node("Identity", ["pixel_values"], ["boxes"])],
            "g",
            [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, list(shape))],
            [helper.make_tensor_value_info("boxes", TensorProto.FLOAT, list(shape))],
        )
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
                  str(self.onnx))

    def _write_meta(self, spec):
        self.onnx.with_suffix(".meta.json").write_text(json.dumps(spec))

    def test_square_meta_gives_a_fully_static_profile(self):
        self._write_meta({"arch": "dfine", "input": {"resize_mode": "square", "size": 640}})
        min_hw, opt_hw, max_hw, source = build_trt.resolve_profile(
            self.onnx, input_name="pixel_values", min_hw=None, opt_hw=None, max_hw=None
        )
        self.assertEqual((min_hw, opt_hw, max_hw), ((640, 640), (640, 640), (640, 640)))
        self.assertIn("meta sidecar", source)

    def test_explicit_flags_win_over_the_meta(self):
        self._write_meta({"arch": "dfine", "input": {"resize_mode": "square", "size": 640}})
        min_hw, opt_hw, max_hw, _ = build_trt.resolve_profile(
            self.onnx, input_name="pixel_values",
            min_hw=(320, 320), opt_hw=(320, 320), max_hw=(320, 320)
        )
        self.assertEqual((min_hw, opt_hw, max_hw), ((320, 320), (320, 320), (320, 320)))

    def test_a_static_graph_shape_is_used_when_there_is_no_meta(self):
        self._write_onnx((1, 3, 512, 512))
        min_hw, opt_hw, max_hw, source = build_trt.resolve_profile(
            self.onnx, input_name="pixel_values", min_hw=None, opt_hw=None, max_hw=None
        )
        self.assertEqual((min_hw, opt_hw, max_hw), ((512, 512), (512, 512), (512, 512)))
        self.assertIn("static ONNX", source)

    def test_a_dynamic_graph_with_no_meta_and_no_flags_is_refused_not_guessed(self):
        with self.assertRaises(SystemExit):
            build_trt.resolve_profile(
                self.onnx, input_name="pixel_values", min_hw=None, opt_hw=None, max_hw=None
            )


class AutoPrecisionPolicyTests(unittest.TestCase):
    """``resolve_auto_precision`` — what ``precision="auto"`` means per arch.

    The interesting case is an arch on the fp16 floor whose AutoCast engine was
    measured clean (dfine): "auto" should reach for that engine when ModelOpt is
    installed and fall back to the fp32 floor when it is not — never to the
    blanket-cast fp16 the floor exists to keep out.
    """

    def test_a_trusted_arch_is_unchanged(self):
        self.assertEqual(
            arch_policy.resolve_auto_precision("rtdetr", modelopt_available=True),
            ("fp16", "graph_cast"))
        self.assertEqual(
            arch_policy.resolve_auto_precision("rtdetr", modelopt_available=False),
            ("fp16", "graph_cast"))

    def test_an_autocast_rescued_arch_uses_autocast_when_modelopt_is_there(self):
        self.assertEqual(
            arch_policy.resolve_auto_precision("dfine", modelopt_available=True),
            ("fp16", "autocast"))

    def test_the_same_arch_falls_back_to_the_fp32_floor_without_modelopt(self):
        precision, _ = arch_policy.resolve_auto_precision("dfine", modelopt_available=False)
        self.assertEqual(precision, "fp32")

    def test_a_floored_arch_with_no_rescue_stays_fp32_either_way(self):
        self.assertEqual(
            arch_policy.resolve_auto_precision("ecdet", modelopt_available=True)[0], "fp32")
        self.assertEqual(
            arch_policy.resolve_auto_precision("ecdet", modelopt_available=False)[0], "fp32")


if __name__ == "__main__":
    unittest.main()
