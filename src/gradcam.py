import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_gradcam(model, img_path, layer_name="block_16_project"):

    # 1. Read image
    img = cv2.imread("/content/data/test_set/test_set/cats/cat.4004.jpg")
    img = cv2.resize(img, (224,224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Preprocess according to MobileNetV2
    img_input = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_rgb.astype(np.float32)
    )
    img_tensor = np.expand_dims(img_input, axis=0)

    # 3. Build grad model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    # 4. Forward + backward pass
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_tensor)
        pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_output)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # 5. Build CAM
    cam = np.zeros(conv_output.shape[1:3], dtype=np.float32)
    conv_output_np = conv_output[0].numpy()
    weights_np = weights.numpy()

    for i, w in enumerate(weights_np):
        cam += w * conv_output_np[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    # 6. Resize CAM
    cam = cv2.resize(cam, (224, 224))

    # 7. Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # 8. Overlay
    superimposed = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)

    return img_rgb, heatmap, superimposed




img_path = "/content/sample.jpg"   # <-- change to your image path

orig, heat, cam = get_gradcam(model, img_path)


plt.figure(figsize=(18,6))

plt.subplot(1,3,1)
plt.imshow(orig)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(heat[..., ::-1])   # BGR → RGB
plt.title("Heatmap")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cam)
plt.title("Grad-CAM Overlay")
plt.axis("off")

plt.savefig("gradcam_output.png", dpi=300)
plt.show()
