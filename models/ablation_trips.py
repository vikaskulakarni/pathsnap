"""Does excluding the 7 train trips with unreliable reference labels help or hurt?

Fits the same model on (a) all 80 train trips and (b) the 73 unflagged trips and compares
them on the 10 student_test trips. Cross-validation is not used here: this is a decision
check on the data, run once with the default configuration.

Run: python -m models.ablation_trips
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from common.config import RESULTS, SEED
from models.common import FEATURE_COLUMNS, chosen_arcs, flagged_train_trips, load_table, point_top1, row_metrics
from models.train_models import connected_share


def main():
    full, test = load_table("train"), load_table("student_test")
    flagged = flagged_train_trips()
    sets = {"all 80 train trips": full, "73 trips (7 flagged excluded)": full[~full.trip_id.isin(flagged)]}
    Xt, yt = test[FEATURE_COLUMNS].to_numpy(float), test.label.to_numpy()
    rows = []
    for name, df in sets.items():
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=SEED)
        m.fit(df[FEATURE_COLUMNS].to_numpy(float), df.label.to_numpy())
        p = m.predict_proba(Xt)[:, 1]
        met = row_metrics(yt, p)
        rows.append({"train_set": name, "trips": df.trip_id.nunique(), "rows": len(df),
                     "test_top1": point_top1(test, p), "test_macro_f1": met["macro_f1"],
                     "test_log_loss": met["log_loss"],
                     "test_route_continuity": connected_share(test, chosen_arcs(test, p))})
    out = pd.DataFrame(rows)
    out.round(5).to_csv(RESULTS / "ablation_flagged_trips.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
