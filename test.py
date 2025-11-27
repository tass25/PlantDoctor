import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers, Model

# Load MobileNet without top layers
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
output = layers.Dense(5, activation='softmax')(x)  # 5 classes, adjust as needed

# Create full model
mobilenet_model = Model(inputs=base_model.input, outputs=output)

# Save as .keras for Keras 3 compatibility
mobilenet_model.save("/workspaces/PlantDoctor/mobilenet_model.keras")

print("✅ MobileNet model wrapped and saved successfully!")
