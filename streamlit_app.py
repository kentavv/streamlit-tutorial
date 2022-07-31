# 🤠🔬

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import scipy
import graphviz as graphviz
import pycm

from scipy.special import jv
from scipy.stats import norm

# st.set_page_config(layout="wide")
st.set_page_config(layout="centered")

# st.title('Streamlit Experiments')
st.title('PR Curves Experiments')
st.write('')

with st.sidebar:
    st.image('https://www.erisyon.com/images/erisyon-logo-5d2e674c.svg')
    st.caption('This app powered by Streamlit, Plotly, PyCM')
    # st.write('The sidebar is shared across tabs.')
    # st.write('Click the X in the upper-right of the sidebar to close')
    # 'Until state management is figured out, press Regenerate All to stop the balloons'
    # )
    # n = st.number_input('Samples', min_value=10, value=100, step=100)
    # mu = st.number_input('Mu', value=10.)
    # std = st.number_input('Std dev', min_value=.01, value=2.)

    # with open('Pipfile') as f:
    #     st.download_button("Download Pipfile", f, file_name='Pipfile')
    # with open('streamlit_app.py') as f:
    #     st.download_button("Download Source", f, file_name='streamlit_app.py')
    #
    # st.button("Regenerate All", key='Regenerate All')

    # if "do_once" not in st.session_state:
    #     st.session_state.do_once = 0
    #     st.write(st.session_state)
    # if st.session_state.do_once == 0:
    #     # st.balloons()
    #     st.session_state.do_once = st.session_state.do_once + 1
    #     st.write('redoing', st.session_state.do_once)
    #     st.write(st.session_state)


def gen_histogram():
    cols = st.columns(3)
    with cols[0]:
        n = st.number_input('Samples', min_value=10, value=100, step=100)
        # Streamlit does not always seem to constrain returned type correctly
        n = max(int(n), 10)
    with cols[1]:
        mu = st.number_input('Mu', value=10., step=1.)
    with cols[2]:
        std = st.number_input('Std dev', min_value=.01, value=2., step=1.)
    # with st.sidebar:
    #     n = st.number_input('Samples', min_value=10, value=100, step=100)
    #     mu = st.number_input('Mu', value=10.)
    #     std = st.number_input('Std dev', min_value=.01, value=2.)
    #     st.button("Regenerate", key='Regenerate Histogram')

    dat = np.random.normal(mu, std, size=n)
    f_mu, f_std = norm.fit(dat)

    fig, ax = plt.subplots()
    ax.hist(dat, bins=20, density=True, label='samples')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    i_p = norm.pdf(x, mu, std)
    f_p = norm.pdf(x, f_mu, f_std)

    ax.plot(x, i_p, 'r', label='underlying')
    ax.plot(x, f_p, 'k', label='observed')
    ax.legend()

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


def gen_network():
    # Create a graphlib graph object
    graph = graphviz.Digraph()
    graph.edge('run', 'intr')
    graph.edge('intr', 'runbl')
    graph.edge('runbl', 'run')
    graph.edge('run', 'kernel')
    graph.edge('kernel', 'zombie')
    graph.edge('kernel', 'sleep')
    graph.edge('kernel', 'runmem')
    graph.edge('sleep', 'swap')
    graph.edge('swap', 'runswap')
    graph.edge('runswap', 'new')
    graph.edge('runswap', 'runmem')
    graph.edge('new', 'runmem')
    graph.edge('sleep', 'runmem')

    st.graphviz_chart(graph)

    st.graphviz_chart('''
        digraph {
            run -> intr
            intr -> runbl
            runbl -> run
            run -> kernel
            kernel -> zombie
            kernel -> sleep
            kernel -> runmem
            sleep -> swap
            swap -> runswap
            runswap -> new
            runswap -> runmem
            new -> runmem
            sleep -> runmem
        }
    ''')


def gen_shell():
    # Streamlit server has no ps, /bin/ps, /usr/bin/ps.
    cmds = ['ls', 'pwd', 'df']
    for cmd in cmds:
        st.header(cmd)
        st.text(os.popen(cmd).read())


def gen_pycm():
    import plotly.express as px
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    import json

    st.write('Select size of two-class population to generate and classify. '
             'Threshold value changes continuous model score to binary prediction with varying Precision and Recall. '
             'Change the Random Seed to generate a new population.')
    cols = st.columns(2)
    with cols[0]:
        n = st.number_input('Samples', min_value=10, value=100000, step=10000)
        n = max(int(n), 10)
        prng_n = st.number_input('Random Seed', min_value=0, value=0, step=1)
        prng_n = max(int(prng_n), 0)
    with cols[1]:
        threshold = st.number_input('Threshold', min_value=0., max_value=1.00001, value=.65, step=.05)
    # st.button("Regenerate", key='Regenerate')

    st.write('')
    st.write('')
    st.write('')

    X, y = make_classification(n_samples=n, random_state=prng_n)

    model = LogisticRegression()
    model.fit(X, y)
    y_score = model.predict_proba(X)[:, 1]

    fpr, tpr, thresholds = roc_curve(y, y_score)

    y_pred = ((y_score > threshold) * 1).astype(int)
    cm = pycm.ConfusionMatrix(y, y_pred, digit=5)

    # fpr_x = np.interp(threshold, np.flipud(thresholds), np.flipud(fpr))
    # tpr_x = np.interp(threshold, np.flipud(thresholds), np.flipud(tpr))
    fpr_x, tpr_x = cm.FPR[1], cm.TPR[1]
    precision_x, recall_x = cm.PPV[1], cm.TPR[1]

    cm2 = cm.matrix

    # 0 is false, 1 is true
    source = [0, 0, 1, 1]
    # 2 is false, 3 is true
    target = [2, 3, 2, 3]
    # uniform
    value = [cm2[0][0], cm2[0][1], cm2[1][0], cm2[1][1]]

    def plot_sankey():
        st.header('Sankey Plot')
        st.write('Through the model, known class samples (left side) are assigned to predicted classes (right side).')

        import plotly.graph_objects as go

        data = go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=["Known False", "Known True", "Predicted False", "Predicted True"],
                color=['#a6cee3', '#fb9a99', '#a6cee3', '#fb9a99'],
            ),

            link=dict(source=source,
                      target=target,
                      value=value,
                      color=['#a6cee3', '#fb9a99', '#a6cee3', '#fb9a99']
                      )
        )

        fig = go.Figure(data)
        # fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)

        # fig.update_layout(
        #     hovermode='x',
        #     title="Sankey Chart",
        #     font=dict(size=10, color='white'),
        # )

        st.plotly_chart(fig)

    def plot_histogram():
        st.header('Histogram')
        st.write('The model has assigned a score to each sample in the population. '
                 'The histogram shows the distribution of scores for False and True classes.')
        st.write('Scores are compared to the threshold (vertical black line) to assign a two-class prediction. '
                 'Samples to the right of the threshold are assigned to class True, and otherwise are assigned to class False.')

        # Sort by the true class so histogram will assign colors to the classes consistently.
        # Histogram seems to always assign colors by the order unique class labels are encountered.
        df = pd.DataFrame({'y': y, 'y_score': y_score}).sort_values(['y', 'y_score'])
        df['y'].replace({0: 'Known False', 1: 'Known True'}, inplace=True)

        fig = px.histogram(
            x=df['y_score'], color=df['y'], nbins=50,
            # labels={'color': 'True Labels', 'x': 'Model Score'},
            labels={'color': '', 'x': 'Model Score'},
            histnorm='percent',
            # marginal="box",  # can be rug, `box`, `violin`
            # cumulative=True
        )

        fig.add_vline(x=threshold)

        st.plotly_chart(fig)

    def plot_threshold_study():
        st.header('Threshold Study')
        st.write('Threshold study of True-Positive Rate and False-Positive Rate.')
        st.write(f'At selected threshold: False-Positive Rate: {fpr_x:.4f}; True-Positive Rate: {tpr_x:.4f}')

        df = pd.DataFrame({
            'False Positive Rate': fpr,
            'True Positive Rate': tpr
        }, index=thresholds)
        df.index.name = "Threshold"
        df.columns.name = "Rate"

        fig = px.line(
            df,
            # title='TPR and FPR at every threshold',
            width=700, height=500
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(range=[0, 1], constrain='domain')

        fig.add_vline(x=threshold)

        st.plotly_chart(fig)

        st.latex(r'\begin{align*}'
                 r'\textrm{True Positive Rate} &= \frac{\left( \textrm{True Positives} \right)}{\left( \textrm{True Positives} + \textrm{False Negatives} \right)} \\'
                 r'&= \textrm{Recall} \\'
                 r'&= \textrm{Sensitivity} \\'
                 r'\textrm{False Positive Rate} &= \frac{\left( \textrm{False Positives} \right)}{\left( \textrm{False Positives} + \textrm{True Negatives} \right)} \\'
                 r'&= 1 - \textrm{Specificity}'
                 r'\end{align*}')

    def plot_pr_curve():
        st.header('Precision-Recall (PR) Curve')
        st.write(f'Area under curve (AUC) = {auc(fpr, tpr):.4f}')
        # st.write(f'At selected threshold: False-Positive Rate: {fpr_x:.4f}; True-Positive Rate: {tpr_x:.4f}')
        st.write(f'At selected threshold: Precision: {precision_x:.4f}; Recall: {recall_x:.4f}')

        precision, recall, thresholds = precision_recall_curve(y, y_score)

        fig = px.area(
            x=recall, y=precision,
            # title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
            labels={'x': 'Recall', 'y': 'Precision'},
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')

        fig.add_annotation(dict(font=dict(color='rgba(0,0,0,0.8)', size=12),
                                x=recall_x,
                                # x = xStart
                                y=precision_x,
                                showarrow=True,
                                text=f'Threshold = {threshold:.2f}',
                                # ax = -10,
                                textangle=0,
                                xanchor='right',
                                xref="x",
                                yref="y"))

        st.plotly_chart(fig)

        st.write(f'The diagonal line in the PR curve is where Precision = Recall.')
        st.write(f'Use a PR Curve when: 1) there is a large imbalance in the dataset since Precision and '
                 'Recall do not depend on True Negatives; or 2) where True Negatives are not a signficant '
                 'concern to the problem objective ')

        st.latex(r'\begin{align*}'
                 r'\textrm{Precision} &= \frac{\left( \textrm{True Positives} \right)}{\left( \textrm{True Positives} + \textrm{False Positives} \right)} \\'
                 r'\textrm{Recall} &= \frac{\left( \textrm{True Positives} \right)}{\left( \textrm{True Positives} + \textrm{False Negatives} \right)} \\'
                 r'\textrm{True Positive Rate} &= \textrm{Recall} \\'
                 r'\textrm{True Positive Rate} &= \textrm{Sensitivity} \\'
                 r'\textrm{False Positive Rate} &= \frac{\left( \textrm{False Positives} \right)}{\left( \textrm{False Positives} + \textrm{True Negatives} \right)}'
                 r'\end{align*}')

    def plot_roc_curve():
        st.header('Receiver Operating Characteristic (ROC) Curve')
        st.write(f'Area under curve (AUC) = {auc(fpr, tpr):.4f}')

        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        fig.add_annotation(dict(font=dict(color='rgba(0,0,0,0.8)', size=12),
                                x=fpr_x,
                                # x = xStart
                                y=tpr_x,
                                showarrow=True,
                                text=f'Threshold = {threshold:.2f}',
                                # ax = -10,
                                textangle=0,
                                xanchor='left',
                                xref="x",
                                yref="y"))

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')

        st.plotly_chart(fig)

        st.write(f'The ROC graph summarizes confusion matrices for a given model for many threshold values.')
        st.write(f'The AUC is a single value summary FPR & TPR tested at many thresholds, and can be used to compare models and model parameters.')
        st.write(f'Find a threshold that maximizes the True-Positive Rate while minimizing the False-Positive to an acceptable level.')
        st.write(f'At selected threshold: False-Positive Rate: {fpr_x:.4f}; True-Positive Rate: {tpr_x:.4f}')
        st.write(f'The diagonal line in the ROC curve is where True-Positive Rate = False-Positive Rate.')

    def plot_cm():
        st.header('Confusion Matrix')
        st.write('Powered by PyCM')

        # st.write(cm.matrix)

        df = pd.DataFrame.from_dict(cm.matrix, orient='index')
        df = df.rename(index={0: 'Known False', 1: 'Known True'}, columns={0: 'Predicted False', 1: 'Predicted True'})
        st.dataframe(df)

        df = pd.DataFrame.from_dict(cm.class_stat, orient='index')
        df[0] = df[0].astype(str)
        df[1] = df[1].astype(str)
        df = df.rename(columns={0: 'Predicted False', 1: 'Predicted True'})
        st.dataframe(df)

        st.text(json.dumps(cm.overall_stat, sort_keys=True, indent=4))

    tabs = [('Sankey', plot_sankey),
            ('Histogram', plot_histogram),
            ('Threshold', plot_threshold_study),
            ('ROC Curve', plot_roc_curve),
            ('PR Curve', plot_pr_curve),
            ('Conf. Mat.', plot_cm)]
    st_tabs = st.tabs([x[0] for x in tabs])

    for i, (tab_name, tab_f) in enumerate(tabs):
        with st_tabs[i]:
            tab_f()


# tabs = [
#         ("PR Curves", gen_pycm),
#         # ("Histogram", gen_histogram),
#         # ("Random Image", gen_random_image),
#         # ("Math", gen_math),
#         # ("PSF", gen_psf),
#         # ("Vega-Lite", gen_vega_chart),
#         # ("Columns", gen_columns),
#         # ("Tables", gen_tables),
#         # ("Network", gen_network),
#         # ("Shell", gen_shell)
#     ]
#
# st_tabs = st.tabs([x[0] for x in tabs])
#
# for i, (tab_name, tab_f) in enumerate(tabs):
#     with st_tabs[i]:
#         st.header(tab_name)
#         tab_f()

gen_pycm()
