import json
import logging

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


RELATIONS_TO_DROP = [
    "['DEATHS_NUMBER', 'IS_OF_SIZE']",
    "['DEATHS_NUMBER', 'INJURED_NUMBER']",
    "['DEATHS_NUMBER', 'INJURED_NUMBER']",
    "['GENDER_MALE', 'GENDER_FEMALE']",
    "['GENDER_MALE', 'IS_IN_CONTACT_WITH']",
    "['INITIATED', 'DIED_IN']",
    "['INITIATED', 'DIED_IN']",
]

NO_RELATION_CLASS = "#NO_RELATION"
RELATION_CLASSES = [
    NO_RELATION_CLASS,
    "END_DATE",
    "GENDER_MALE",
    "WEIGHS",
    "DIED_IN",
    "HAS_FAMILY_RELATIONSHIP",
    "IS_DEAD_ON",
    "IS_IN_CONTACT_WITH",
    "HAS_CATEGORY",
    "HAS_CONTROL_OVER",
    "IS_BORN_IN",
    "IS_OF_SIZE",
    "HAS_LATITUDE",
    "IS_PART_OF",
    "IS_OF_NATIONALITY",
    "IS_COOPERATING_WITH",
    "DEATHS_NUMBER",
    "HAS_FOR_HEIGHT",
    "INITIATED",
    "WAS_DISSOLVED_IN",
    "HAS_COLOR",
    "CREATED",
    "IS_LOCATED_IN",
    "WAS_CREATED_IN",
    "IS_AT_ODDS_WITH",
    "HAS_CONSEQUENCE",
    "HAS_FOR_LENGTH",
    "INJURED_NUMBER",
    "START_DATE",
    "STARTED_IN",
    "GENDER_FEMALE",
    "HAS_LONGITUDE",
    "RESIDES_IN",
    "HAS_FOR_WIDTH",
    "IS_BORN_ON",
    "HAS_QUANTITY",
    "OPERATES_IN",
    "IS_REGISTERED_AS",
]
ENTITY_CLASSES = [
    "TERRORIST_OR_CRIMINAL",
    "LASTNAME",
    "LENGTH",
    "NATURAL_CAUSES_DEATH",
    "COLOR",
    "STRIKE",
    "DRUG_OPERATION",
    "HEIGHT",
    "INTERGOVERNMENTAL_ORGANISATION",
    "TRAFFICKING",
    "NON_MILITARY_GOVERNMENT_ORGANISATION",
    "TIME_MIN",
    "DEMONSTRATION",
    "TIME_EXACT",
    "FIRE",
    "QUANTITY_MIN",
    "MATERIEL",
    "GATHERING",
    "PLACE",
    "CRIMINAL_ARREST",
    "CBRN_EVENT",
    "ECONOMICAL_CRISIS",
    "ACCIDENT",
    "LONGITUDE",
    "BOMBING",
    "MATERIAL_REFERENCE",
    "WIDTH",
    "FIRSTNAME",
    "MILITARY_ORGANISATION",
    "CIVILIAN",
    "QUANTITY_MAX",
    "CATEGORY",
    "POLITICAL_VIOLENCE",
    "EPIDEMIC",
    "TIME_MAX",
    "TIME_FUZZY",
    "NATURAL_EVENT",
    "SUICIDE",
    "CIVIL_WAR_OUTBREAK",
    "POLLUTION",
    "ILLEGAL_CIVIL_DEMONSTRATION",
    "NATIONALITY",
    "GROUP_OF_INDIVIDUALS",
    "QUANTITY_FUZZY",
    "RIOT",
    "WEIGHT",
    "THEFT",
    "MILITARY",
    "NON_GOVERNMENTAL_ORGANISATION",
    "LATITUDE",
    "COUP_D_ETAT",
    "ELECTION",
    "HOOLIGANISM_TROUBLEMAKING",
    "QUANTITY_EXACT",
    "AGITATING_TROUBLE_MAKING",
]


def format_relations_str_to_list(labels_as_str: str) -> list[str]:
    return (
        json.loads(labels_as_str.replace("{", "[").replace("}", "]").replace("'", '"'))
        if not pd.isnull(labels_as_str)
        else [NO_RELATION_CLASS]
    )


# def encode_target_to_onehot(
#     data: pd.DataFrame,
#     labels_as_list_column: str,
#     onehot_label_encoder: MultiLabelBinarizer,
# ) -> pd.DataFrame:
#     assert hasattr(
#         onehot_label_encoder, "classes_"
#     ), "Fit the onehot_label_encoder` first!"
#     onehot_columns = onehot_label_encoder.classes_
#     data[onehot_columns] = pd.DataFrame(
#         onehot_label_encoder.transform(
#             data[labels_as_list_column].apply(format_relations_str_to_list)
#         ),
#         columns=onehot_columns,
#         index=data.index,
#     ).astype(float)
#     return data


def encode_target_to_onehot(
    data: pd.DataFrame,
    labels_as_list_column: str,
    onehot_label_encoder: MultiLabelBinarizer,
) -> pd.DataFrame:
    assert hasattr(
        onehot_label_encoder, "classes_"
    ), "Fit the onehot_label_encoder` first!"
    onehot_columns = onehot_label_encoder.classes_
    logging.info(f"{onehot_columns=}")
    data.loc[:, onehot_columns] = onehot_label_encoder.transform(
        data[labels_as_list_column].apply(format_relations_str_to_list)
    ).astype(float)
    return data
