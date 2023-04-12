import numpy as np
import torch
from scipy.stats import hmean
from sklearn.metrics import classification_report, roc_auc_score


def classification_metrics(samples_df, position_df, verbose=True):
    str_labels = sorted([str(t) for t in position_df.label.unique()])
    target = position_df[position_df.label == 1].diagnosis.unique()
    diag_count = position_df.groupby("diagnosis").size()
    selected_diagnoses = diag_count[diag_count >= 10].index.values

    # 1. Sample Metrics
    sample_labels = samples_df.label.values
    sample_predictions = samples_df.prediction.values
    sample_report = classification_report(
        sample_labels, sample_predictions, output_dict=True, zero_division=0
    )
    metrics = {f"{t}_sample_recall": sample_report[t]["recall"] for t in str_labels}
    if verbose:
        print("\nSample Classification Report\n")
        print(classification_report(sample_labels, sample_predictions, zero_division=0))

    # Next
    position_df["prediction"] = (position_df["output"].values > 0.5).astype(int)
    pos_score = {}
    to_print = []

    # 3. Disease Statistics
    for pos in sorted(position_df.position.unique()):
        selection = position_df[position_df.position == pos]
        diagnoses = selection.diagnosis.values

        patient_predictions = []
        for diagnosis, prediction in zip(diagnoses, selection.prediction.values):
            if diagnosis in target:
                if prediction:
                    patient_predictions.append(diagnosis)
                else:
                    wrong_prediction = int(target[0] == 0)  # 1 if healthy, 0 otherwise
                    patient_predictions.append(wrong_prediction)
            else:
                if prediction:
                    wrong_prediction = target[0]
                    patient_predictions.append(wrong_prediction)
                else:
                    patient_predictions.append(diagnosis)

        patient_predictions = np.array(patient_predictions)

        pos_disease_report = classification_report(
            diagnoses, patient_predictions, output_dict=True, zero_division=0
        )
        # should replace by roc-auc?

        pos_recall = []
        neg_recall = []
        for diagnosis in selected_diagnoses:  # sorted(position_df.diagnosis.unique())
            diagnosis_recall = pos_disease_report[str(diagnosis)]["recall"]
            if diagnosis in target:  # changeme
                pos_recall.append(diagnosis_recall)
            else:
                neg_recall.append(diagnosis_recall)

        pos_recall = np.array(pos_recall).mean()
        neg_recall = np.array(neg_recall).mean()

        score = hmean([pos_recall, neg_recall])
        pos_score[pos] = score

        to_print.append(
            (pos, ["{:.2f}".format(v) for v in [neg_recall, pos_recall, score]])
        )

    # 4. ROC-AUC scores: Added
    labels = position_df.label.values
    logits = position_df["output"].values
    roc_auc = roc_auc_score(labels, logits)
    print("ROC-AUC Score:", roc_auc)

    for pos in sorted(position_df.position.unique()):
        selection = position_df[position_df.position == pos]
        labels = selection.label.values
        logits = selection.output.values
        pos_roc_auc = roc_auc_score(labels, logits)
        print(pos, "ROC-AUC Score:", pos_roc_auc)
        pos_score[pos] = pos_roc_auc

    # 5. print
    if verbose:
        print(
            "\nHarmonic mean of selected class recalls, selected classes:",
            selected_diagnoses,
        )
        print("Weighted Class Recalls:")
        row_format = "{:>10}" * (len(str_labels) + 2)
        print(row_format.format("", *(str_labels + ["Score"])))
        for pos, ls in to_print:
            print(row_format.format(pos, *ls))
        print()

    return metrics, pos_score


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    epoch,
    device,
    scheduler=None,
    log_interval=10,
):
    model.train()

    total_batch_loss = 0

    for batch_idx, batch_dict in enumerate(train_loader):  # added
        data = batch_dict["data"].to(device)
        target = batch_dict["target"].to(device)

        optimizer.zero_grad()

        # ➡ Forward pass
        output_dict = model(data)

        output = output_dict["diagnosis_output"]
        loss = criterion(output, target)
        total_batch_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        if scheduler is not None:  # added
            scheduler.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {:3d} [{:6d}/{:6d} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    train_loss = total_batch_loss / len(train_loader)
    return train_loss


def evaluate(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0

    val_samples_df = val_loader.dataset.samples_df
    val_samples_df["prediction"] = -1

    n_classes = len(val_samples_df.label.unique())

    position_df = val_samples_df[
        ["patient", "diagnosis", "position", "label"]
    ].drop_duplicates()
    output_cols = [f"output_{target}" for target in range(n_classes)]
    for col in output_cols:
        position_df[col] = 0.0
    position_df = position_df.set_index(["patient", "position"])

    # Added
    position_df["output"] = 0.0

    with torch.no_grad():
        for batch_dict in val_loader:  # added
            # Load the input features and labels from the val dataset
            sample_idx = batch_dict["sample_idx"]
            data = batch_dict["data"].to(device)
            target = batch_dict["target"].to(device)

            # Make predictions: Pass image data from val dataset, make predictions about class image belongs to
            output_dict = model(data)
            output = output_dict["diagnosis_output"]
            pos_output = output

            # Compute the loss sum up batch loss
            batch_size = data.shape[0]
            val_loss += batch_size * criterion(output, target).item()

            # Add sample predictions: CHANGEME
            # val_samples_df.loc[sample_idx.cpu().numpy(), "prediction"] = output.cpu().numpy().argmax(axis=-1)

            # Add position outputs
            patient = val_samples_df.at[sample_idx[0].item(), "patient"]
            position = val_samples_df.at[sample_idx[0].item(), "position"]

            # Added
            position_df.at[(patient, position), "output"] = pos_output.cpu().numpy()[
                0, 0
            ]

    position_df = position_df.reset_index()
    metrics, pos_score = classification_metrics(
        val_samples_df, position_df, verbose=True
    )
    val_loss = val_loss / len(val_samples_df)
    metrics["Validation Loss"] = val_loss

    return metrics, pos_score
