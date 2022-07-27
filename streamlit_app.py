import numpy as np
import streamlit as st

st.title('Yo')
st.write(np.zeros((3,3)))

img = np.random.randint(255, size=(100, 100), dtype=np.uint8)
st.image(img, caption='Random image', output_format='PNG')
