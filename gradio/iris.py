import gradio as gr

# prevent from running on every reload (will run twice when starting watch mode)
if gr.NO_RELOAD:
    import argparse
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import seaborn as sns

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="Iris")
    args, unknown = parser.parse_known_args()

    # load data
    iris = load_iris()
    X, y, feature_names, target_names = (
        iris.data,
        iris.target,
        iris.feature_names,
        iris.target_names,
    )

    # convert iris to dataframe for visualizing (unscaled, string labels)
    iris_df = pd.DataFrame(
        data=np.c_[X, y],
        columns=np.append(feature_names, ["target"]),
    )
    iris_df.target = iris_df.target.apply(lambda x: target_names[int(x)])

    # normalize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3D projection
    start_pca = time.time()
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)
    fig_3d = px.scatter_3d(
        X_pca,
        x=0,
        y=1,
        z=2,
        color=[target_names[int(i)] for i in y],
        color_discrete_sequence=px.colors.qualitative.D3,
        labels={"color": "Species", "0": "x", "1": "y", "2": "z"},
    )
    print(f"PCA finished in {time.time() - start_pca:.4f}s")

    # train model
    # (in real-world, load from pre-trained)
    start_train = time.time()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    print(f"Training finished in {time.time() - start_train:.4f}s")

description = """
The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken from Fisher's paper. Note that it's the same as in R, but not as in the UCI Machine Learning Repository, which has two wrong data points.\n
This is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.)\n
The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
"""

references = """
- Fisher, R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to Mathematical Statistics" (John Wiley, NY, 1950).
- Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis. (Q327.D83) John Wiley & Sons. ISBN 0-471-22361-1. See page 218.
- Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System Structure and Classification Rule for Recognition in Partially Exposed Environments". IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. PAMI-2, No. 1, 67-71.
- Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule". IEEE Transactions on Information Theory, May 1972, 431-433.
- See also: 1988 MLC Proceedings, 54-64. Cheeseman et al"s AUTOCLASS II conceptual clustering system finds 3 classes in the data.
- Many, many more ...
"""


# use synthetic delay to demonstrate progress
def on_predict_click(sepal_length, sepal_width, petal_length, petal_width, progress=gr.Progress()):
    # 0/100
    progress(0, total=100, desc="Predicting...")

    inputs = [[sepal_length, sepal_width, petal_length, petal_width]]
    inputs = scaler.transform(inputs)

    # 1..99/100
    for i in range(1, 100):
        progress(i / 100, total=100, desc="Predicting...")
        time.sleep(np.random.uniform(0.01, 0.03))

    probas = model.predict_proba(inputs)
    probas = np.squeeze(probas)

    # 100/100
    progress(1, total=100, desc="Predicting...")
    return {name: proba for (name, proba) in zip(target_names, probas)}


def on_plot_change(x, y):
    # figures created through `plt.figure()` are kept in memory until explicitly closed
    plt.close()
    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=iris_df,
        x=x,
        y=y,
        hue="target",
        palette=sns.color_palette("tab10", 3),
    )
    return fig


with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown(f"# {args.title}\n\n{description}")

    with gr.Tab("Model"):
        with gr.Row():
            with gr.Column():
                model_select = gr.Dropdown(
                    ["Logistic Regression"],
                    value="Logistic Regression",
                    label="Classifier",
                )

                sepal_length_slider = gr.Slider(
                    iris_df[feature_names[0]].min(),
                    iris_df[feature_names[0]].max(),
                    label=feature_names[0],
                    step=0.1,
                )
                sepal_width_slider = gr.Slider(
                    iris_df[feature_names[1]].min(),
                    iris_df[feature_names[1]].max(),
                    label=feature_names[1],
                    step=0.1,
                )
                petal_length_slider = gr.Slider(
                    iris_df[feature_names[2]].min(),
                    iris_df[feature_names[2]].max(),
                    label=feature_names[2],
                    step=0.1,
                )
                petal_width_slider = gr.Slider(
                    iris_df[feature_names[3]].min(),
                    iris_df[feature_names[3]].max(),
                    label=feature_names[3],
                    step=0.1,
                )

                predict_button = gr.Button("Predict", variant="primary")

            with gr.Column():
                predict_label = gr.Label({"unknown": 1.0}, label="Species")

                # inputs are the arguments to `fn`
                # output components receive the return value of `fn`
                predict_button.click(
                    fn=on_predict_click,
                    inputs=[
                        sepal_length_slider,
                        sepal_width_slider,
                        petal_length_slider,
                        petal_width_slider,
                    ],
                    outputs=[predict_label],
                    api_name="predict",
                )

        with gr.Row():
            # caching examples creates a `gradio_cached_examples` folder
            gr.Examples(
                fn=on_predict_click,
                run_on_click=True,
                cache_examples=False,
                examples=[
                    [5.1, 3.5, 1.4, 0.2],  # setosa
                    [7.0, 3.2, 4.7, 1.4],  # versicolor
                    [6.3, 3.3, 6.0, 2.5],  # virginica
                ],
                inputs=[
                    sepal_length_slider,
                    sepal_width_slider,
                    petal_length_slider,
                    petal_width_slider,
                ],
                outputs=[predict_label],
            )

    with gr.Tab("Data"):
        with gr.Row():
            gr.DataFrame(value=iris_df)

        with gr.Accordion(label="Feature Comparison", open=True):
            with gr.Row():
                with gr.Column(scale=2):
                    # the Plot component lets you use custom plots
                    scatter_plot = gr.Plot(
                        value=on_plot_change(feature_names[0], feature_names[1]),
                        format="png",
                        show_label=False,
                    )

                with gr.Column(scale=1):
                    x_select = gr.Dropdown(
                        feature_names,
                        value=feature_names[0],
                        label="X-axis",
                        filterable=False,
                    )
                    y_select = gr.Dropdown(
                        feature_names,
                        value=feature_names[1],
                        label="Y-axis",
                        filterable=False,
                    )

                    # hide plotting functions from API
                    x_select.change(
                        fn=on_plot_change,
                        inputs=[x_select, y_select],
                        outputs=[scatter_plot],
                        show_api=False,
                    )
                    y_select.change(
                        fn=on_plot_change,
                        inputs=[x_select, y_select],
                        outputs=[scatter_plot],
                        show_api=False,
                    )

        with gr.Accordion(label="Principal Component Analysis", open=True):
            gr.Plot(value=fig_3d, show_label=False)

        with gr.Accordion(label="Summary statistics", open=False):
            gr.Markdown(iris_df.describe().to_markdown())

        with gr.Accordion(label="References", open=False):
            gr.Markdown(references)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=None)
    demo.launch(
        debug=True,
        share=False,
        server_port=7860,
        server_name="0.0.0.0",
    )
