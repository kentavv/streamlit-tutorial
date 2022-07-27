import numpy as np
import streamlit as st
import scipy
from scipy.special import jv
import matplotlib.pyplot as plt

st.title('Streamlit Experiments')

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)
st.pyplot(fig)

# def gen():
#     img = np.random.randint(low=slider[0], high=slider[1], size=(100, 100), dtype=np.uint8)
#     st.image(img, caption='Random image', output_format='PNG')

st.write(np.zeros((3, 3)))
st.write(np.zeros((3, 3)))
slider = st.slider('Pixel range', min_value=0, max_value=255, value=(25, 225), step=1) #, on_change=gen)

if st.button('Generate random image'):
    pass
    # gen()

img = np.random.randint(low=slider[0], high=slider[1], size=(100, 100), dtype=np.uint8)
st.image(img, caption='Random image', output_format='PNG')

st.latex(r'''
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')

fig, ax = plt.subplots()

x = np.linspace(-10, 10, 1000)
for i in range(0, 2 + 1):
    plt.plot(x, jv(i, x), label=f'$J_{i}(x)$')
st.pyplot(fig)


fig, ax = plt.subplots()
slider = st.slider('J', min_value=-4, max_value=4, step=1) #, on_change=gen)
x = np.linspace(-20, 20, 1000)
i = slider
plt.plot(x, jv(i, x), label=f'$J_{i}(x)$')
st.pyplot(fig)
