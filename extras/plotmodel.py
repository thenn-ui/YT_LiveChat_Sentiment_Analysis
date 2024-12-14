from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Load the model
model = load_model("sentiment_analysis_model.h5")

# Plot the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
