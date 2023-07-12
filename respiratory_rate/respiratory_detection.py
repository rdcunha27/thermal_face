import numpy as np
import insightface

import respiratory_config as config

# detection_model = insightface.model_zoo.get_model('retinaface_mnet025_v1')
detection_model = insightface.model_zoo.get_model('buffalo_l/det_10g.onnx')
detection_model.prepare(ctx_id=config.GPU_ID, nms=0.4)


def get_face_detection(image, width, height):
    """ Get face detection result from given image.

    Args:
        image: An numpy array with shape `(height, width, 3)`.

    Returns:
        The bounding boxes and landmarks of the detected faces.
    """
    # duplicated_image = np.stack([image] * 3, -1)
    bounding_boxes, landmarks = detection_model.detect(
        image,
        input_size = [height, width]
        # input_size = [448, 448]
        # threshold=config.FACE_DETECTION_THRESHOLD,
        # scale=1.0
    )
    return bounding_boxes.astype(int)[:, :-1], landmarks.astype(int)
