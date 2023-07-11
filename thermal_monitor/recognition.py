import insightface
import cv2

embedding_model = insightface.model_zoo.get_model('buffalo_l/w600k_r50.onnx')
# embedding_model = insightface.model_zoo.get_model('retinaface_mnet025_v1')
embedding_model.prepare(ctx_id=-1)


def get_embedding(face_image):
    """ Returns the embedding vector for a given face image.

    Args:
        face_image: A numpy array with shape `(height, width, 3)`.
    """
    reshaped_image = cv2.resize(face_image, (112, 112))
    return embedding_model.get_embedding(reshaped_image)[0]
