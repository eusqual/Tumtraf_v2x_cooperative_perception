import json
import os

import numpy as np
from collections import defaultdict

IOU_THRESH = 0.1


reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'


# -------------------------------------------------
# Geometry
# -------------------------------------------------
def aabb_iou_3d(box1, box2):
    """
    Axis-aligned IoU for 3D boxes.
    box = [x,y,z,sx,sy,sz]
    """

    def bounds(b):
        x,y,z,sx,sy,sz = b
        return np.array([
            x - sx/2, y - sy/2, z - sz/2,
            x + sx/2, y + sy/2, z + sz/2
        ])

    b1 = bounds(box1)
    b2 = bounds(box2)

    inter_min = np.maximum(b1[:3], b2[:3])
    inter_max = np.minimum(b1[3:], b2[3:])
    inter = np.maximum(inter_max - inter_min, 0)

    inter_vol = inter[0]*inter[1]*inter[2]

    vol1 = np.prod(b1[3:] - b1[:3])
    vol2 = np.prod(b2[3:] - b2[:3])

    union = vol1 + vol2 - inter_vol
    return inter_vol / union if union > 0 else 0


# -------------------------------------------------
# JSON loader
# -------------------------------------------------
def load_openlabel(path):
    data = json.load(open(path))

    frames = []

    for frame in data["openlabel"]["frames"].values():
        objs = frame.get("objects", {})
        frame_boxes = []

        for obj in objs.values():
            cub = obj["object_data"]["cuboid"]
            vals = cub["val"]
            cls = obj["object_data"]["type"]

            score = 1.0
            for a in cub.get("attributes", {}).get("num", []):
                if a["name"] == "score":
                    score = a["val"]

            x,y,z,qx,qy,qz,qw,sx,sy,sz = vals

            frame_boxes.append({
                "class": cls,
                "box": [x,y,z,sx,sy,sz],
                "score": score
            })

        frames.append(frame_boxes)

    return frames


# -------------------------------------------------
# Matching
# -------------------------------------------------
def match_frame(gt, pred, iou_thresh):
    """
    returns TP, FP, FN per class
    """

    results = defaultdict(lambda: [0,0,0])  # TP FP FN

    classes = set([g["class"] for g in gt] + [p["class"] for p in pred])

    for cls in classes:
        gts = [g for g in gt if g["class"]==cls]
        preds = sorted([p for p in pred if p["class"]==cls],
                       key=lambda x: -x["score"])

        matched = set()

        for p in preds:
            best_iou = 0
            best_idx = -1

            for i,g in enumerate(gts):
                if i in matched: continue
                iou = aabb_iou_3d(p["box"], g["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_thresh:
                results[cls][0] += 1
                matched.add(best_idx)
            else:
                results[cls][1] += 1

        results[cls][2] += len(gts) - len(matched)

    return results


# -------------------------------------------------
# Metrics
# -------------------------------------------------
def compute_metrics(all_results):
    class_stats = defaultdict(lambda: np.zeros(3))

    for frame_res in all_results:
        for cls,vals in frame_res.items():
            class_stats[cls] += vals

    per_class = {}

    for cls,(tp,fp,fn) in class_stats.items():
        prec = tp/(tp+fp) if tp+fp>0 else 0
        rec  = tp/(tp+fn) if tp+fn>0 else 0
        ap   = prec * rec   # simplified AP proxy

        per_class[cls] = dict(
            precision=prec,
            recall=rec,
            AP=ap
        )

    mAP = np.mean([v["AP"] for v in per_class.values()]) if per_class else 0

    return per_class, mAP


# -------------------------------------------------
# Main evaluation
# -------------------------------------------------
def evaluate(gt_path, pred_path):

    gt_frames   = load_openlabel(gt_path)
    pred_frames = load_openlabel(pred_path)

    assert len(gt_frames)==len(pred_frames), "Frame count mismatch"

    all_results = []

    for gt,pred in zip(gt_frames, pred_frames):
        all_results.append(match_frame(gt,pred,IOU_THRESH))

    per_class, mAP = compute_metrics(all_results)

    print("\nPer-class results")
    for cls,v in per_class.items():
        print(cls, v)

    print("\nOverall mAP:", mAP)

    scores = {
        "3d_map": mAP,
    }
    # write scores to scores.json
    with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
        score_file.write(json.dumps(scores))


# -------------------------------------------------
if __name__ == "__main__":
    debugging = False
    if debugging:
        # use for local testing
        evaluate("test_annotations_testsplit.json", "submission.json")
    else:
        # use on submission server
        evaluate(os.path.join(reference_dir, "test_annotations_testsplit.json"), os.path.join(prediction_dir, "submission.json"))

   