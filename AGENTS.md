# Repository Agent Notes

## Unified SAM Backend Goal

Keep SAM-related Label Studio behavior under one ML backend server. The backend should support both interactive SAM click segmentation and text-prompt Grounding-SAM preannotation through the same `/predict` endpoint.

### Desired Runtime Behavior

- If `context` is present, run interactive point-click SAM.
  - Label Studio smart keypoint input sends the click location.
  - Backend returns a `PolygonLabels` region on the `segmentation` control.
    (The export pipeline only handles bbox + polygon, so SAM vectorizes its
    mask to a polygon rather than emitting a brush mask. Brush output is still
    available behind `SAM_OUTPUT=brush`, but is no longer the default.)
- If `context` is missing and custom params include `grounding_sam`, run Grounding-SAM autoannotation.
  - Example custom params: `{"grounding_sam": "person, helmet, vest"}`.
  - Parse comma/newline-separated classes or prompts.
  - Run the Grounding-SAM package over the requested task images and return predictions.
- If `context` is missing and no `grounding_sam` param is present, return an empty prediction.
  - Opening an image with autoannotation enabled may call `/predict` with `context: null`; that should not trigger expensive model work.

### Intended Dispatch Shape

```python
def predict(self, tasks, context=None, **kwargs):
    params = kwargs.get("params") or {}

    if context:
        return self.predict_sam_point(tasks, context=context)

    if params.get("grounding_sam"):
        return self.predict_grounding_sam(
            tasks,
            prompt=params["grounding_sam"],
        )

    return [empty_prediction(self.model_version)]
```

### Label Studio Config

Use smart keypoint input for SAM prompts and polygon output for masks:

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>

  <RectangleLabels name="bbox" toName="image">
    <Label value="Object"/>
  </RectangleLabels>

  <PolygonLabels name="segmentation" toName="image">
    <Label value="Object"/>
  </PolygonLabels>

  <KeyPointLabels name="sam_point" toName="image" smart="true">
    <Label value="Object" smart="true" showInline="true"/>
  </KeyPointLabels>
</View>
```

The annotator workflow is: select smart `sam_point`, click the object, receive
a polygon on the `segmentation` control, then optionally edit the polygon
vertices for cleanup. Brush is intentionally not offered — the COCO/YOLO export
pipeline has no mask format (see `coco_sync/` and `fleet.py sync`).

### Implementation Plan

1. Use `ml_backends/sam` as the single SAM backend server.
2. Keep the current point-click SAM behavior working without custom params.
3. Add a Grounding-SAM path activated only by custom params such as `grounding_sam`.
4. Load heavy models lazily so opening images with no params remains cheap.
5. Share image-loading logic across both paths.
6. Ensure `SAM_DATA_ROOT + task path` resolves inside the backend container.
7. Keep Docker as one container on port `9090`.
8. Add clear logs for dispatch decisions: empty, `sam_point`, or `grounding_sam`.
9. Test three cases: open task with no params returns empty, smart keypoint click returns a polygon, and `grounding_sam` params return project/task preannotations.
