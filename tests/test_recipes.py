from opentq.recipes import get_recipe


def test_qwen36_recipe_contains_balanced_v2_release() -> None:
    recipe = get_recipe("qwen3.6-27b")
    releases = {release.slug for release in recipe.releases}
    assert "Qwen3.6-27B-TQ4_BAL_V2" in releases
    assert "Qwen3.6-27B-TQ4_SB2" in releases
    assert "Qwen3.6-27B-TQ4_SB4" in releases
