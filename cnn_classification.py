########## 1. Import required libraries ##########
import argparse
import os
import re

import numpy as np
import pandas as pd
from gensim.models import Phrases
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import NearMiss
from keras import backend
from keras import callbacks
from keras import layers
from keras import metrics
from keras import optimizers
from keras import Sequential
from keras_tuner import Hyperband
from nltk import download
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


########## 2. Downloads and setup ##########

# Download NLTK utils
download("stopwords")
download("punkt_tab")
download("wordnet")

# Create a set of english stopwords
stop_words = set(stopwords.words("english"))

# Intiialise a lemmatizer
lemmatizer = WordNetLemmatizer()

embeddings_directory = "embeddings"
results_directory = "results_cnn"

WORD2VEC = "word2vec"
PRETRAINED_GLOVE = "pretrained_glove"
PRETRAINED_FASTTEXT = "pretrained_fasttext"
VALID_EMBEDDINGS = [WORD2VEC, PRETRAINED_GLOVE, PRETRAINED_FASTTEXT]

########## 3. Define preprocess methods ##########


def pre_process(text: str) -> str:
    # Convert all text to lowercase
    text = text.lower()

    # Remove all HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove all URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Nltk tokenize (split) the text
    words = word_tokenize(text)

    # Remove stopwords and lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Join words back into a single string
    text = " ".join(words)

    # Only keep alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text


def combine_title_body(row):
    if pd.notna(row["Body"]):
        return row["Title"] + ". " + row["Body"]

    return row["Title"]


########## 4. Define embedding methods ##########


def get_word2vec_embeddings(
    vectorizer: layers.TextVectorization,
    sentences,
    vocabulary_size: int,
    embedding_dimension: int,
):
    word2vec = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5)

    embedding_matrix = np.zeros((vocabulary_size, embedding_dimension))
    for index, word in enumerate(vectorizer.get_vocabulary()):
        if index < vocabulary_size:
            if word in word2vec.wv:
                embedding_matrix[index] = word2vec.wv[word]

    return embedding_matrix


def embedding_matrix_from_vectorizer(
    vectorizer: layers.TextVectorization,
    embeddings: dict,
    vocabulary_size: int,
    embedding_dimension: int,
) -> np.ndarray:
    embedding_matrix = np.zeros((vocabulary_size, embedding_dimension))

    for index, word in enumerate(vectorizer.get_vocabulary()):
        if index < vocabulary_size:
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    return embedding_matrix


def get_glove_embeddings(
    vectorizer: layers.TextVectorization,
    vocabulary_size: int,
    path: str | None,
):
    if path is None:
        path = "glove.6B.300d.txt"

    embeddings = {}
    with open(f"{embeddings_directory}/{path}", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector

    embedding_dimension = len(embeddings[next(iter(embeddings))])

    embedding_matrix = embedding_matrix_from_vectorizer(
        vectorizer,
        embeddings,
        vocabulary_size,
        embedding_dimension,
    )
    return embedding_matrix, embedding_dimension


def get_fasttext_embeddings(
    vectorizer: layers.TextVectorization,
    vocabulary_size: int,
    path: str | None,
):
    if path is None:
        path = "wiki-news-300d-1M.vec"

    embeddings = {}
    with open(f"{embeddings_directory}/{path}", encoding="utf8") as f:
        _, embedding_dimension = map(int, next(f).split())
        for line in f:
            values = line.split()
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[values[0]] = vector

    embedding_matrix = embedding_matrix_from_vectorizer(
        vectorizer,
        embeddings,
        vocabulary_size,
        embedding_dimension,
    )
    return embedding_matrix, embedding_dimension


########## 5. Define models ##########


def model_builder(
    hp,
    vocabulary_size: int,
    embedding_dim: int,
    embedding_matrix,
) -> Sequential:
    """Defines a model that can be tuned with hyperparameter tuning"""

    hp_conv_layers = hp.Int("conv_layers", min_value=1, max_value=2, step=1)
    hp_dense_layers = hp.Int("dense_layers", min_value=1, max_value=2, step=1)

    model = Sequential()

    model.add(
        layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False,
        ),
    )

    for conv_layers in range(hp_conv_layers):
        hp_filters = hp.Int(
            f"filters+{conv_layers}",
            min_value=32,
            max_value=128,
            step=32,
        )
        hp_kernel = hp.Int(f"kernel+{conv_layers}", min_value=1, max_value=5, step=1)

        model.add(
            layers.Conv1D(filters=hp_filters, kernel_size=hp_kernel, activation="relu"),
        )

        if conv_layers == hp_conv_layers - 1:
            model.add(layers.GlobalMaxPooling1D())
        else:
            model.add(layers.MaxPooling1D(pool_size=2))

    for dense_layer in range(hp_dense_layers):
        hp_units = hp.Int(f"units+{dense_layer}", min_value=32, max_value=128, step=32)
        hp_dropout = hp.Float(
            f"dropout+{dense_layer}",
            min_value=0.2,
            max_value=0.7,
            step=0.1,
        )

        model.add(layers.Dense(units=hp_units, activation="relu"))
        model.add(layers.Dropout(hp_dropout))

    model.add(layers.Dense(1, activation="sigmoid"))

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.003,
        decay_steps=1000,
        decay_rate=0.90,
        staircase=True,
    )

    optimizer = optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", metrics.Precision(), metrics.Recall(), metrics.AUC()],
    )

    return model


def pre_tuned_model(
    vocabulary_size: int,
    embedding_dim: int,
    embedding_matrix: np.ndarray,
) -> Sequential:
    """A model that has been manually tuned for TensorFlow dataset"""
    model = Sequential()

    model.add(
        layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False,
        ),
    )

    model.add(layers.Conv1D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(units=64, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.003,
        decay_steps=1000,
        decay_rate=0.90,
        staircase=True,
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", metrics.Precision(), metrics.Recall(), metrics.AUC()],
    )

    return model


########## 6. Training and Evaluation ##########
def main(
    dataset: str,
    embeddings: str,
    pretrained_embedding_path: str | None,
    manual_tuned_model: bool,
    repetitions: int,
    generate_hyperparameters: bool,
) -> None:
    vocabulary_size = 10000
    output_sequence_length = 200

    # --- 6.1 Read data ---
    df = pd.read_csv(f"datasets/{dataset}.csv")

    # --- 6.1 Read data ---

    df["text"] = df.apply(combine_title_body, axis=1)

    processed_text = "processed_text"

    df[processed_text] = df["text"].apply(pre_process)

    if embeddings == "pretrained-fasttext":
        phrases = Phrases(df[processed_text], min_count=1, threshold=1)
        phraser = Phraser(phrases)

        df[processed_text].apply(lambda text: phraser[text])

    ########## 8. Vectorize text ##########

    vectorizer = layers.TextVectorization(
        standardize=None,
        max_tokens=vocabulary_size,
        output_sequence_length=output_sequence_length,
    )
    vectorizer.adapt(df[processed_text])

    x_data = vectorizer(df[processed_text])

    y_data = df["class"].values

    ########## 8. Create embedding matrix and define embedding dimension ##########

    if embeddings == WORD2VEC:
        embedding_dimension = 100
        embedding_matrix = get_word2vec_embeddings(
            vectorizer,
            df["cleaned_text"],
            vocabulary_size,
            embedding_dimension,
        )
    elif embeddings == PRETRAINED_GLOVE:
        embedding_matrix, embedding_dimension = get_glove_embeddings(
            vectorizer,
            vocabulary_size,
            pretrained_embedding_path,
        )
    elif embeddings == PRETRAINED_FASTTEXT:
        embedding_matrix, embedding_dimension = get_fasttext_embeddings(
            vectorizer,
            vocabulary_size,
            pretrained_embedding_path,
        )
    else:
        raise Exception("No embedding specified")

    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    auc_values = []

    best_hps = None

    for repeat in range(repetitions):
        # Clear memory between runs
        backend.clear_session()

        # --
        x_train, x_test, y_train, y_test = train_test_split(
            x_data.numpy(),
            y_data,
            test_size=0.2,
            shuffle=True,
        )

        undersampler = NearMiss(version=1)
        batch_size = 32
        sequence = BalancedBatchGenerator(
            x_train,
            y_train,
            sampler=undersampler,
            batch_size=batch_size,
        )

        if manual_tuned_model:
            model = pre_tuned_model(
                vocabulary_size,
                embedding_dimension,
                embedding_matrix,
            )
        else:
            tuner = Hyperband(
                lambda hp: model_builder(
                    hp,
                    vocabulary_size,
                    embedding_dimension,
                    embedding_matrix,
                ),
                objective="val_accuracy",
                directory=f"tuners/{dataset}/{embedding_dimension}",
                project_name="hyperband",
                max_epochs=100,
            )

            if (
                generate_hyperparameters and repeat == 0
            ):  # Only need to run the hyperparameter tuning once
                tuner_early_stopping = callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    verbose=1,
                    restore_best_weights=True,
                )
                tuner.search_space_summary()
                tuner.search(
                    sequence,
                    epochs=100,
                    validation_data=(x_test, y_test),
                    callbacks=[tuner_early_stopping],
                )

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(best_hps)

        # Setup early stopping so that it restores weights if the loss does not decrease after 5 iterations
        fit_early_stopping = callbacks.EarlyStopping(
            monitor="loss",
            patience=5,
            verbose=1,
            restore_best_weights=True,
        )

        # Fit the model using 100 epochs and early stopping
        model.fit(
            x_train,
            y_train,
            epochs=100,
            batch_size=32,
            callbacks=[fit_early_stopping],
        )

        y_pred = model.predict(x_test)
        y_label = (y_pred > 0.5).astype(int)

        # Macro average accuracy
        accuracy = accuracy_score(y_test, y_label)
        accuracy_values.append(accuracy)

        # Macro average precision
        precision = precision_score(y_test, y_label, average="macro")
        precision_values.append(precision)

        # Macro average recall
        recall = recall_score(y_test, y_label, average="macro")
        recall_values.append(recall)

        # Macro average f1-score
        f1 = f1_score(y_test, y_label, average="macro")
        f1_values.append(f1)

        # AUC
        x, y, _ = roc_curve(y_test, y_pred, pos_label=1)
        auc_value = auc(x, y)
        auc_values.append(auc_value)

    average_accuracy = np.mean(accuracy_values)
    average_precision = np.mean(precision_values)
    average_recall = np.mean(recall_values)
    average_f1 = np.mean(f1_values)
    average_auc = np.mean(auc_values)

    print("=== CNN Results ===")
    print(f"Number of Repetitions: {repetitions}")
    print(f"Average Accuracy: {average_accuracy:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Average Recall: {average_recall:.4f}")
    print(f"Average F1: {average_f1:.4f}")
    print(f"Average AUC: {average_auc:.4f}")

    if manual_tuned_model:
        final_results_directory = os.path.join(results_directory, "manual")
    else:
        final_results_directory = os.path.join(results_directory, "hyperband")

    os.makedirs(final_results_directory, exist_ok=True)

    results_path = f"{final_results_directory}/{dataset}.csv"

    results_df = pd.DataFrame(
        {
            "Repetition": range(repetitions),
            "Accuracy": accuracy_values,
            "Precision": precision_values,
            "Recall": recall_values,
            "F1": f1_values,
            "AUC": auc_values,
        },
    )

    results_df.to_csv(
        results_path,
        mode="a",
        header=(not os.path.exists(results_path)),
        index=False,
    )

    averages_path = f"{final_results_directory}/averages.csv"

    average_df = pd.DataFrame(
        {
            "Dataset": [dataset],
            "Accuracy": [average_accuracy],
            "Precision": [average_precision],
            "Recall": [average_recall],
            "F1": [average_f1],
            "AUC": [average_auc],
        },
    )

    average_df.to_csv(
        averages_path,
        mode="a",
        header=(not os.path.exists(averages_path)),
        index=False,
    )

    print("=== Model Information ===")
    print(f"Embedding Dimension: {embedding_dimension}")
    print(
        f"Embeddings: {embeddings}",
    )

    if best_hps is not None:
        print("\nConv Layers")
        for conv_layer in range(best_hps.get("conv_layers")):
            print(
                f" -- Layer-{conv_layer}: Filters: {best_hps.get(f'filters+{conv_layer}')}  Kernel: {best_hps.get(f'kernel+{conv_layer}')}",
            )

        print("\nDense Layers")
        for dense_layer in range(best_hps.get("dense_layers")):
            print(
                f" -- Layer-{dense_layer}: Units: {best_hps.get(f'units+{dense_layer}')}  Dropout: {best_hps.get(f'dropout+{dense_layer}')}",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="CNN-Bug-Report-Classification", formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--dataset",
        default="tensorflow",
        help="Dataset to use (e.g. tensorflow/keras)",
    )
    parser.add_argument(
        "--pretrained-embedding-path",
        help="Path to custom pretrained embeddings",
    )
    parser.add_argument(
        "--manual-tuned-model",
        action="store_true",
        help="Use a manually tuned module, instead of hyperparameter tuning",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Specify the number of repetitions",
    )
    parser.add_argument(
        "--generate-hyperparameters",
        action="store_true",
        help="Overwrite, and (re)run Hyperband to search for optimal hyperparameters (not recommended)",
    )
    parser.add_argument('--embeddings',
                        default="pretrained_fasttext",
                        choices=VALID_EMBEDDINGS,
                        help="Chose an embedding")


    args = parser.parse_args()

    main(
        dataset=args.dataset,
        embeddings=args.embeddings,
        pretrained_embedding_path=args.pretrained_embedding_path,
        manual_tuned_model=args.manual_tuned_model,
        repetitions=args.repetitions,
        generate_hyperparameters=args.generate_hyperparameters,
    )
