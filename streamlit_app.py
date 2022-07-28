# 🤠🔬

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import scipy

from scipy.special import jv
from scipy.stats import norm

# st.set_page_config(layout="wide")
st.set_page_config(layout="centered")

st.title('Streamlit Experiments')

with st.sidebar:
    st.image('https://www.erisyon.com/images/erisyon-logo-5d2e674c.svg')
    st.write('The sidebar is shared across tabs.'
             'Click the X in the upper-right of the sidebar to close')
    # n = st.number_input('Samples', min_value=10, value=100, step=100)
    # mu = st.number_input('Mu', value=10.)
    # std = st.number_input('Std dev', min_value=.01, value=2.)
    st.button("Regenerate All", key='Regenerate All')


def gen_histogram():
    cols = st.columns(3)
    with cols[0]:
        n = st.number_input('Samples', min_value=10, value=100, step=100)
    with cols[1]:
        mu = st.number_input('Mu', value=10.)
    with cols[2]:
        std = st.number_input('Std dev', min_value=.01, value=2.)
    # with st.sidebar:
    #     n = st.number_input('Samples', min_value=10, value=100, step=100)
    #     mu = st.number_input('Mu', value=10.)
    #     std = st.number_input('Std dev', min_value=.01, value=2.)
    #     st.button("Regenerate", key='Regenerate Histogram')

    dat = np.random.normal(mu, std, size=n)
    f_mu, f_std = norm.fit(dat)

    fig, ax = plt.subplots()
    ax.hist(dat, bins=20, density=True)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    i_p = norm.pdf(x, mu, std)
    f_p = norm.pdf(x, f_mu, f_std)

    ax.plot(x, i_p, 'r')
    ax.plot(x, f_p, 'k')

    st.pyplot(fig)

    st.write('Graph X range:', xmin, xmax)
    st.write(pd.DataFrame(dat).describe().transpose())


def gen_random_image():
    # def gen():
    #     img = np.random.randint(low=slider[0], high=slider[1], size=(100, 100), dtype=np.uint8)
    #     st.image(img, caption='Random image', output_format='PNG')

    slider = st.slider('Pixel range', min_value=0, max_value=255, value=(25, 225), step=1)  # , on_change=gen)

    st.button('Regenerate', key='Regenerate Image')

    img = np.random.randint(low=slider[0], high=slider[1], size=(100, 100), dtype=np.uint8)
    st.image(img, caption='Random image', output_format='PNG', use_column_width='always')


def gen_math():
    st.latex(r'''
         a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
         \sum_{k=0}^{n-1} ar^k =
         a \left(\frac{1-r^{n}}{1-r}\right)
         ''')


def gen_psf():
    st.write('SciPy version:', scipy.__version__)

    fig, ax = plt.subplots()

    x = np.linspace(-10, 10, 1000)
    for i in range(0, 2 + 1):
        plt.plot(x, jv(i, x), label=f'$J_{i}(x)$')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    slider = st.slider('J', min_value=-4, max_value=4, step=1)  # , on_change=gen)
    x = np.linspace(-20, 20, 1000)
    i = slider
    plt.plot(x, jv(i, x), label=f'$J_{i}(x)$')
    st.pyplot(fig)

    #
    # df1 = pd.DataFrame(
    #     np.random.randn(50, 20),
    #     columns=('col %d' % i for i in range(20)))
    #
    # my_table = st.table(df1)
    #
    # df2 = pd.DataFrame(
    #     np.random.randn(50, 20),
    #     columns=('col %d' % i for i in range(20)))
    #
    # my_table.add_rows(df2)
    # # Now the table shown in the Streamlit app contains the data for
    # # df1 followed by the data for df2.
    #
    # # Assuming df1 and df2 from the example above still exist...
    # my_chart = st.line_chart(df1)
    # my_chart.add_rows(df2)
    # # Now the chart shown in the Streamlit app contains the data for
    # # df1 followed by the data for df2.
    #
    # my_chart = st.vega_lite_chart({
    #      'mark': 'line',
    #      'encoding': {'x': 'a', 'y': 'b'},
    #      'datasets': {
    #        'some_fancy_name': df1,  # <-- named dataset
    #       },
    #      'data': {'name': 'some_fancy_name'},
    #  }),
    # my_chart.add_rows(some_fancy_name=df2)  # <-- name used as keyword


def gen_vega_chart():
    df = pd.DataFrame(
        np.random.randn(200, 3),
        columns=['a', 'b', 'c'])

    st.vega_lite_chart(df, {
        'mark': {'type': 'circle', 'tooltip': True},
        'encoding': {
            'x': {'field': 'a', 'type': 'quantitative'},
            'y': {'field': 'b', 'type': 'quantitative'},
            'size': {'field': 'c', 'type': 'quantitative'},
            'color': {'field': 'c', 'type': 'quantitative'},
        },
    })

    x = np.linspace(-2 * np.pi, -2 * np.pi, 20)
    y = np.sin(x)
    c = [0]
    df = pd.DataFrame(
        [[x, y, c]],
        columns=['x', 'y1', 'c'])

    st.vega_lite_chart(df, {
        'mark': {'type': 'line', 'tooltip': True},
        'encoding': {
            'x': {'field': 'x', 'type': 'quantitative'},
            'y1': {'field': 'y1', 'type': 'quantitative'},
            'color': {'field': 'c', 'type': 'quantitative'},
        },
    })


def gen_columns():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg")
        st.header("A cat")
        st.image("https://static.streamlit.io/examples/cat.jpg")
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg")

    with col2:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg")
        st.header("A cat")
        st.image("https://static.streamlit.io/examples/cat.jpg")

    with col3:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg")


def gen_tables():
    x = np.linspace(-2 * np.pi, -2 * np.pi, 20)
    y = np.sin(x)
    c = [0]
    df = pd.DataFrame(
        [[x, y, c]],
        columns=['x', 'y1', 'c'])
    st.write('Dataframe')
    st.dataframe(df)
    st.write('Table')
    st.table(df)
    st.write('Write')
    st.write(df)


tabs = [("Histogram", gen_histogram),
        ("Random Image", gen_random_image),
        ("Math", gen_math),
        ("PSF", gen_psf),
        ("Vega", gen_vega_chart),
        ("Columns", gen_columns),
        ("Tables", gen_tables)]

st_tabs = st.tabs([x[0] for x in tabs])

for i, (tab_name, tab_f) in enumerate(tabs):
    with st_tabs[i]:
        st.header(tab_name)
        tab_f()
