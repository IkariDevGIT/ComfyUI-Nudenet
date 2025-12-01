# BBOX output schema and ApplyNudenet design

## Impact Pack BBOX/SEGS expectations
- Impact Pack detectors (for example `ONNXDetector.detect`) return a `SEGS` tuple shaped as `((height, width), [segments...])`, where each segment is the `SEG` namedtuple `('cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper')`.
- Bounding boxes inside `SEG` entries are 4-tuples `(x1, y1, x2, y2)` derived from detection results and combined with a cropped mask that fills the detected box (optionally dilated) and the crop region calculated via `utils.make_crop_region(...)`.
- Downstream nodes such as `BboxDetectorForEach`, `BboxDetectorCombined`, and `SimpleDetectorForEach` expect `BBOX_DETECTOR` providers to yield this `SEGS` structure so they can hand detections to detailers, SAM refiners, and mask combiners without additional translation.

## Current Nudenet detections
- `postprocess` currently emits a list of dictionaries shaped like `{"id": class_id, "score": float_score, "box": [left, top, width, height]}` after ONNX inference and NMS filtering, using width/height extents instead of corner coordinates.
- `nudenet_execute` consumes that list to censor images but does not expose any detection output to the ComfyUI graph.

## Proposed ApplyNudenet outputs and packaging
- Expand `ApplyNudenet.RETURN_TYPES` to include both the censored image and a `SEGS` object, e.g. `RETURN_TYPES = ("IMAGE", "SEGS")` with `RETURN_NAMES = ("image", "detections")` to follow ComfyUI multi-output naming conventions.
- Build the `SEGS` payload as `((height, width), segments)` where each segment converts the Nudenet box `[x, y, w, h]` to `(x1, y1, x2, y2)`, populates `confidence` from the detection `score`, and sets `label` to the class name via `CLASSIDS_LABELS_MAPPING` for parity with Impact Pack detectors.
- For each detection, synthesize a binary `cropped_mask` that fills the detected box within the computed crop region (reusing the Impact Pack pattern of filling the detected rectangle) so downstream consumers receive a familiar mask layout. Leave `cropped_image` and `control_net_wrapper` as `None` like the Impact Pack ONNX detector when no extra data is available.

## Workflow compatibility checks
- Using `SEGS` keeps the output interoperable with Impact Pack nodes that already branch on that type, allowing Nudenet detections to plug into detailer and mask workflows without bespoke adapters.
- Providing `RETURN_NAMES` and returning a tuple `(output_image, segs)` matches ComfyUI conventions for nodes with multiple outputs, ensuring the workflow JSON stores the outputs predictably and maintains backward compatibility for the existing censored-image output.
