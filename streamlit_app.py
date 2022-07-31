import streamlit as st
import pycm
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import json

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def gen_page():
    global n, prng_n, threshold, p_class_err

    # st.set_page_config(layout="wide")
    st.set_page_config(layout="centered")

    st.title('PR Curves Experiments')

    with st.sidebar:
        st.image('https://www.erisyon.com/images/erisyon-logo-5d2e674c.svg')
        st.caption('This app powered by Streamlit, Plotly, PyCM')

        st.write('Select size of two-class population to generate and classify. '
                 'Threshold value is used to change continuous model scores to binary class predictions. '
                 'ROC and PR curves summarize the performance of the model for all threshold values. '
                 'Performance of the model for a specific threshold is derived from the confusion matrix. '
                 'Change the Random Seed to generate a new population.')

        n = st.number_input('Samples', min_value=10, value=100000, step=10000)
        n = max(int(n), 10)
        prng_n = st.number_input('Random Seed', min_value=0, value=0, step=1)
        prng_n = max(int(prng_n), 0)
        p_class_err = st.number_input('Probability of class error', min_value=0., max_value=1.00001, value=0.01, step=.05)

        threshold = st.number_input('Threshold', min_value=0., max_value=1.00001, value=.65, step=.05)


def gen_pycm():
    X, y = make_classification(n_samples=n, random_state=prng_n, flip_y=p_class_err)

    # model = LogisticRegression(penalty='elasticnet', l1_ratio=l1_ratio, solver='saga')
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
        st.write(f'At selected threshold: False-Positive Rate: {fpr_x:.4f}; True-Positive Rate: {tpr_x:.4f}')

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
        st.write(f'The diagonal line in the ROC curve is where True-Positive Rate = False-Positive Rate.')

    def plot_cm():
        st.header('Confusion Matrix')

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


gen_page()
gen_pycm()
