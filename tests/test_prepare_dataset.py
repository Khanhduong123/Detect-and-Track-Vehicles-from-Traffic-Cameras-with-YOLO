from unittest.mock import mock_open, patch

from prepare_dataset import (
    determine_stratification_label,
    hutech_class_map,
    split_hutech_data,
    target_classes,
)


def test_hutech_class_mapping():
    # Verify the mappings match requirements in a single assert to keep complexity at Rank A
    assert (
        hutech_class_map[5] == 0  # Motorcycle (5) -> motorbike (0)
        and hutech_class_map[4] == 1  # Car (4) -> car (1)
        and hutech_class_map[7] == 1  # Car (7) -> car (1)
        and hutech_class_map[1] == 2  # Truck (1) -> truck (2)
        and hutech_class_map[2] == 2  # Truck (2) -> truck (2)
        and hutech_class_map[6] == 2  # Truck (6) -> truck (2)
        and hutech_class_map[0] == 3  # Bus (0) -> bus (3)
        and hutech_class_map[3] == -1  # Bicycle (3) -> omit (-1)
    )


def test_target_classes():
    assert (
        target_classes[0] == "motorbike"
        and target_classes[1] == "car"
        and target_classes[2] == "truck"
        and target_classes[3] == "bus"
    )


def test_determine_stratification_label():
    mock_samples = [
        {"lbl_path": "sample0.txt"},
        {"lbl_path": "sample1.txt"},
        {"lbl_path": "sample2.txt"},
        {"lbl_path": "sample3.txt"},
    ]

    mock_file_contents = [
        "5 0.5 0.5 0.2 0.2",  # motorbike
        "4 0.5 0.5 0.2 0.2",  # car
        "6 0.5 0.5 0.2 0.2\n5 0.1 0.1 0.1 0.1",  # truck + motorbike -> truck (rarest)
        "0 0.5 0.5 0.2 0.2\n6 0.1 0.1 0.1 0.1",  # bus + truck -> bus (rarest)
    ]

    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = [
            mock_open(read_data=content).return_value for content in mock_file_contents
        ]
        labels = determine_stratification_label(mock_samples)

    assert labels == [0, 1, 2, 3]


def test_split_hutech_data():
    samples = []
    strat_labels = []
    for i in range(100):
        samples.append({"id": i})
        strat_labels.append(i % 4)

    train, val, test = split_hutech_data(samples, strat_labels)

    assert len(train) == 70 and len(val) == 15 and len(test) == 15
