from src.reports import load_reports, normalize_report, placeholder_signature


def test_intact_repair_is_specific():
    text = "intact9xintact4cm; the ACL is intact"
    normalized = normalize_report(text)

    assert "0.9x0.4cm" in normalized
    assert "the ACL is intact" in normalized


def test_report_loading_adds_placeholder_signature(synthetic_dataset):
    reports = load_reports(synthetic_dataset / "train.csv")
    first_report = reports.iloc[0]

    assert "0.9x0.4cm" in first_report["Report"]
    assert first_report["PlaceholderSignature"] == "none"


def test_placeholder_signature_preserves_present_tokens():
    signature = placeholder_signature("[DATE] at [TIME], [ID]")

    assert signature == "[DATE]+[TIME]+[ID]"
