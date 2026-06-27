import os
import sys
from pathlib import Path
from urllib.parse import unquote

from flask import jsonify, request

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from label_studio_ml.api import _manager, init_app
from model import NewModel


app = init_app(
    model_class=NewModel,
    model_dir=os.getenv("MODEL_DIR", "/tmp/label-studio-ml"),
)


def route_params(prompt=None):
    if prompt is None:
        return {}
    return {"grounding_sam": unquote(prompt)}


def setup_without_job_lookup(prompt=None):
    data = request.json or {}
    params = route_params(prompt)
    print(
        "SAMBackend setup "
        f"keys={sorted(data.keys())} project={data.get('project')} url_params={params}",
        flush=True,
    )
    model = _manager.fetch(
        data.get("project"),
        data.get("schema"),
        data.get("force_reload", False),
        hostname=data.get("hostname", ""),
        access_token=data.get("access_token", ""),
        model_version=None,
    )
    model.model_version = model.model.model_version
    return jsonify({"model_version": model.model_version})


def predict_with_lazy_setup(prompt=None):
    data = request.json or {}
    tasks = data.get("tasks")
    project = data.get("project")
    label_config = data.get("label_config")
    force_reload = data.get("force_reload", False)
    try_fetch = data.get("try_fetch", True)
    params = {**route_params(prompt), **(data.get("params") or {})}
    params_context = params.pop("context", None)
    context = data.get("context") if data.get("context") is not None else params_context

    if not _manager._current_model:
        _manager.fetch(project, label_config, False, model_version=None)

    predictions, model = _manager.predict(
        tasks,
        project,
        label_config,
        force_reload,
        try_fetch,
        context=context,
        **params,
    )
    model.model_version = model.model.model_version
    return jsonify({
        "results": predictions,
        "model_version": model.model_version,
    })


def ignore_training_webhook(prompt=None):
    data = request.json or {}
    return jsonify({
        "status": "ignored",
        "reason": "SAM backend does not train from Label Studio webhooks.",
        "action": data.get("action"),
    }), 201


def prefixed_health(prompt=None):
    return jsonify({
        "status": "UP",
        "model_dir": _manager.model_dir,
        "v2": os.getenv("LABEL_STUDIO_ML_BACKEND_V2", default=False),
        "url_params": route_params(prompt),
    })


app.view_functions["_setup"] = setup_without_job_lookup
app.view_functions["_predict"] = predict_with_lazy_setup
app.view_functions["webhook"] = ignore_training_webhook

app.add_url_rule(
    "/grounding_sam=<path:prompt>/health",
    endpoint="grounding_sam_health",
    view_func=prefixed_health,
    methods=["GET"],
)
app.add_url_rule(
    "/grounding_sam=<path:prompt>/",
    endpoint="grounding_sam_root",
    view_func=prefixed_health,
    methods=["GET"],
)
app.add_url_rule(
    "/grounding_sam=<path:prompt>/setup",
    endpoint="grounding_sam_setup",
    view_func=setup_without_job_lookup,
    methods=["POST"],
)
app.add_url_rule(
    "/grounding_sam=<path:prompt>/predict",
    endpoint="grounding_sam_predict",
    view_func=predict_with_lazy_setup,
    methods=["POST"],
)
app.add_url_rule(
    "/grounding_sam=<path:prompt>/webhook",
    endpoint="grounding_sam_webhook",
    view_func=ignore_training_webhook,
    methods=["POST"],
)


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "9090")),
        debug=os.getenv("DEBUG", "false").lower() == "true",
    )
