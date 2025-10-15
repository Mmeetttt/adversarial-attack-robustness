import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- App Configuration ---
# Set the page title and icon for the browser tab
st.set_page_config(page_title="Adversarial Attack Demo", page_icon="🛡️")

# --- Load Resources (Cached) ---
# Use st.cache_resource to load the model and data only once, speeding up the app.
@st.cache_resource
def load_resources():
    """Loads the pre-trained model and the Fashion-MNIST dataset."""
    model = tf.keras.models.load_model('fashion_mnist_cnn.h5')
    (_, _), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    test_images = test_images / 255.0
    test_images = test_images.reshape((10000, 28, 28, 1))
    return model, test_images, test_labels

model, test_images, test_labels = load_resources()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# --- Adversarial Attack Function ---
# This is the same FGSM function we developed earlier.
def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

# --- Streamlit UI ---
st.title("🛡️ Adversarial Attack on a CNN")
st.write("""
This app demonstrates how a Convolutional Neural Network (CNN) trained on the Fashion-MNIST dataset can be fooled by an adversarial attack.
The attack (FGSM) adds a tiny, human-imperceptible layer of noise to an image to cause a misclassification.
""")

st.markdown("---")

# --- User Input ---
st.sidebar.header("Attack Configuration")
# Slider for the user to select an image from the test set
selected_index = st.sidebar.slider("Select an image index from the test set:", 0, 9999, 42)
# Slider for the user to control the attack strength (epsilon)
epsilon = st.sidebar.slider("Attack Strength (Epsilon):", 0.0, 0.3, 0.1, 0.01)

# --- Display Original Image and Prediction ---
st.header("1. Original Image")
original_image = test_images[selected_index:selected_index+1]
original_label = test_labels[selected_index]
original_label_as_array = test_labels[selected_index:selected_index+1] # Get label as a NumPy slice

# Display the image
st.image(original_image.reshape(28, 28), caption=f"Original Image: A '{class_names[original_label]}'", width=200)

# Predict with the model
pred_probs = model.predict(original_image)
pred_label_index = np.argmax(pred_probs)
confidence = np.max(pred_probs) * 100

st.subheader(f"Prediction: `{class_names[pred_label_index]}`")
st.write(f"Confidence: **{confidence:.2f}%**")
if pred_label_index == original_label:
    st.success("The model's initial prediction is correct! ✅")
else:
    st.warning("The model's initial prediction is incorrect. ❌")

st.markdown("---")

# --- Perform and Display Attack ---
st.header("2. The Attack")
if st.button("🚀 Attack this Image!"):
    st.write("Generating adversarial image...")

    # Generate the adversarial image
    original_image_tensor = tf.convert_to_tensor(original_image)
    # *** FIX: Pass the label as a NumPy array slice, not a Python list ***
    perturbations = create_adversarial_pattern(original_image_tensor, original_label_as_array)
    adversarial_image = original_image_tensor + epsilon * perturbations
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

    # Predict on the adversarial image
    adv_pred_probs = model.predict(adversarial_image)
    adv_pred_label_index = np.argmax(adv_pred_probs)
    adv_confidence = np.max(adv_pred_probs) * 100

    st.subheader("Attack Results")

    col1, col2 = st.columns(2)
    with col1:
        st.image(adversarial_image.numpy().reshape(28, 28), caption="Adversarial Image (Looks the same!)", width=200)
    with col2:
        st.subheader(f"New Prediction: `{class_names[adv_pred_label_index]}`")
        st.write(f"Confidence: **{adv_confidence:.2f}%**")
        if adv_pred_label_index != original_label:
            st.error("The attack was successful! The model was fooled. 💥")
        else:
            st.info("The attack failed. The model's prediction did not change. 💪")

    st.write(f"The adversarial image, which looks identical to the original, has tricked the model into changing its prediction from **'{class_names[original_label]}'** to **'{class_names[adv_pred_label_index]}'**.")

