from opentq.recipes import get_recipe


def test_qwen36_recipe_contains_balanced_dense_release() -> None:
    recipe = get_recipe("qwen3.6-27b")
    releases = {release.slug for release in recipe.releases}
    assert "Qwen3.6-27B-TQ_BAL_DENSE" in releases
    assert "Qwen3.6-27B-TQ4_SB4" in releases
