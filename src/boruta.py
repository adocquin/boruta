"""
Boruta algorithm implementation.
Based on the original paper:
https://www.jstatsoft.org/article/view/v036i11

The Boruta algorithm consists of following steps:
1. Extend the information system by adding copies of all variables
(the information system is always extended by at least 5 shadow
attributes, even if the number of attributes in the original set is
lower than 5).
2. Shuffle the added attributes to remove their correlations with the
response.
3. Run a random forest classifier on the extended information system and
gather the Z scores computed.
4. Find the maximum Z score among shadow attributes (MZSA), and then
assign a hit to every attribute that scored better than MZSA.
5. For each attribute with undetermined importance perform a two-sided
test of equality with the MZSA.
6. Deem the attributes which have importance significantly lower than
MZSA as ‘unimportant’ and permanently remove them from the information
system.
7. Deem the attributes which have importance significantly higher than
MZSA as ‘important’.
4 Feature Selection with the Boruta Package
8. Remove all shadow attributes.
9. Repeat the procedure until the importance is assigned for all the
attributes, or the algorithm has reached the previously set limit of the
random forest runs.

In practice this algorithm is preceded with three start-up rounds, with
less restrictive importance criteria. The startup rounds are introduced
to cope with high fluctuations of Z scores when the number of attributes
is large at the beginning of the procedure. During these initial rounds,
attributes are compared respectively to the fifth, third and second best
shadow attribute; the test for rejection is performed only at the end
of each initial round, while the test for confirmation is not performed
at all.
"""
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import binom


class Decision(Enum):
    UNDETERMINED: int = 0
    IMPORTANT: int = 1
    UNIMPORTANT: int = -1


def boruta(
    x_original: pd.DataFrame, y: pd.Series, max_round: int = 100, alpha=0.05
) -> dict[str, list[str]]:
    # Step 0: Initialize variables
    x: pd.DataFrame = x_original.copy()
    hits: dict[str, int] = {k: 0 for k in x.columns}
    decisions: dict[str, Decision] = {k: Decision.UNDETERMINED for k in x.columns}

    for round in range(max_round):
        # Step 1: Extend the information system by adding copies of
        # all variables
        x_extended: pd.DataFrame = pd.concat([x, x], axis=1, keys=["", "shadow_"])
        # The information system is always extended by at least 5 shadow
        # attributes, even if the number of attributes in the original
        # set is lower than 5)
        n_features: int = x.shape[1]
        for i in range(5 - n_features):
            x_extended[f"added_shadow_{i}"] = np.random.permutation(
                x_extended.iloc[:, i].values
            )

        # Step 2: Shuffle the values of the shadow features
        x_extended.iloc[:, n_features:] = x_extended.iloc[:, n_features:].apply(
            np.random.permutation
        )

        # Step 3: Run a random forest classifier on the extended
        # information system and gather the Z scores computed.
        random_forest: RandomForestClassifier = RandomForestClassifier(
            random_state=np.random.randint(0, 1000)
        )
        random_forest.fit(x_extended, y)
        feature_importances: dict[str, float] = {
            k: v for k, v in zip(x.columns, random_forest.feature_importances_)
        }
        shadow_importances: np.ndarray = random_forest.feature_importances_[n_features:]

        # Step 4: Find the maximum Z score among shadow attributes
        # (MZSA), and then assign a hit to every attribute that scored
        # better than MZSA.
        if round > 2:
            zsa: float = np.max(shadow_importances)
        # Step 4.1: Apply correction for first 3 rounds
        else:
            sorted_shadow_importances: np.ndarray = np.sort(shadow_importances)[::-1]
            if round == 0:
                zsa = sorted_shadow_importances[4]
            elif round == 1:
                zsa = sorted_shadow_importances[2]
            else:
                zsa = sorted_shadow_importances[1]
        hits = {key: hits[key] + 1 if value > zsa else hits[key] for key, value in feature_importances.items()}

        # Step 5: For each attribute with undetermined importance
        # perform a two-sided test of equality with the MZSA.
        # Step 5.1: Determine confirmed and rejected criteria.
        unimportant_threshold: float = binom.ppf(alpha, round + 1, 0.5)
        important_threshold: float = binom.ppf(1 - alpha, round + 1, 0.5)
        for feature in x.columns:
            if decisions[feature] == Decision.UNDETERMINED:
                # Step 6: Deem the attributes which have importance
                # significantly lower than MZSA as 'unimportant' and
                # permanently remove them from the information system.
                if hits[feature] < unimportant_threshold:
                    decisions[feature] = Decision.UNIMPORTANT
                    x.drop(feature, axis=1, inplace=True)
                # Deem the attributes which have importance
                # significantly higher than MZSA as 'important'.
                elif round > 2 and hits[feature] > important_threshold:
                    decisions[feature] = Decision.IMPORTANT

        # Step 8. Remove all shadow attributes.
        del x_extended

        # Step 9. Repeat the procedure until the importance is assigned
        # for all the attributes, or the algorithm has reached the
        # previously set limit of the random forest runs.
        if all(value != Decision.UNDETERMINED for value in decisions.values()):
            break

    result: dict[str, list[str]] = {
        "important": [k for k in decisions.keys() if decisions[k] == Decision.IMPORTANT],
        "unimportant": [k for k in decisions.keys() if decisions[k] == Decision.UNIMPORTANT],
        "undetermined": [k for k in decisions.keys() if decisions[k] == Decision.UNDETERMINED],
    }
    return result