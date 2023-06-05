def get_model_file(
    model_name: str, target_str: str, val_fold: int, test_fold: int
) -> str:
    res = "{}_D{}_V{}_T{}.pt".format(model_name, target_str, val_fold, test_fold)
    return res


def get_output_file(
    model_name: str, target_str: str, val_fold: int, test_fold: int
) -> str:
    res = "{}_outputs_D{}_V{}_T{}.npy".format(
        model_name, target_str, val_fold, test_fold
    )
    return res


def get_attention_file(
    model_name: str, target_str: str, val_fold: int, test_fold: int
) -> str:
    res = "{}_attn_D{}_V{}_T{}.npy".format(model_name, target_str, val_fold, test_fold)
    return res


def get_aggregate_file(
    model_name: str, target_str: str, val_fold: int, test_fold: int
) -> str:
    res = "{}_agg_D{}_V{}_T{}.csv".format(model_name, target_str, val_fold, test_fold)
    return res
