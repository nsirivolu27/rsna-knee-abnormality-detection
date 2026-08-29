import pandas as pd

from src.site_proxy import build_site_proxy, normalize_manufacturer


def test_manufacturer_normalization_preserves_unknowns():
    assert normalize_manufacturer("GE Medical Systems") == "ge"
    assert normalize_manufacturer("Toshiba") == "canon"
    assert normalize_manufacturer("Acme Scanner") == "acme_scanner"


def test_small_groups_are_coarsened_and_auditable():
    metadata = pd.DataFrame(
        {
            "StudyInstanceUID": [str(i) for i in range(6)],
            "Manufacturer": ["Siemens"] * 6,
            "ManufacturerModelName": [f"Model-{i}" for i in range(6)],
            "MagneticFieldStrength": [1.5 + i / 10 for i in range(6)],
        }
    )
    reports = pd.DataFrame(
        {
            "StudyInstanceUID": [str(i) for i in range(6)],
            "detected_language": ["en"] * 6,
            "placeholder_signature": [f"sig-{i}" for i in range(6)],
        }
    )

    result = build_site_proxy(metadata, reports, min_group_size=3)

    assert result["site_proxy_assignment_level"].eq("manufacturer_language").all()
    assert result["site_proxy_group_size"].eq(6).all()
    assert result["site_proxy_key"].map(len).eq(2).all()
    assert result["full_site_proxy_key"].map(len).eq(5).all()


def test_language_fallback_remains_visible_when_still_small():
    metadata = pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b"],
            "Manufacturer": ["Siemens", "Philips"],
            "ManufacturerModelName": ["A", "B"],
            "MagneticFieldStrength": [1.5, 3.0],
        }
    )
    reports = pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b"],
            "detected_language": ["en", "de"],
            "placeholder_signature": ["x", "y"],
        }
    )

    result = build_site_proxy(metadata, reports, min_group_size=3)

    assert result["site_proxy_assignment_level"].eq("language").all()
    assert result["site_proxy_under_minimum"].all()
    assert result["site_proxy_key"].map(len).eq(1).all()
