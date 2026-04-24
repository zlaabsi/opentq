from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReleaseArtifact:
    slug: str
    weight_profile: str
    profile_type: str
    target_size_gib: str
    quality_band: str
    mac_target: str
    runtimes: tuple[str, ...]
    priority: int
    notes: str


@dataclass(frozen=True)
class WorkPhase:
    name: str
    goal: str
    deliverables: tuple[str, ...]
    exit_criteria: tuple[str, ...]


@dataclass(frozen=True)
class ModelRecipe:
    key: str
    model_id: str
    family: str
    architecture: str
    context_length: str
    base_weight_size_gib: str
    releases: tuple[ReleaseArtifact, ...]
    phases: tuple[WorkPhase, ...]


QWEN36_27B_RECIPE = ModelRecipe(
    key="qwen3.6-27b",
    model_id="Qwen/Qwen3.6-27B",
    family="dense-hybrid-vlm",
    architecture="64-layer hybrid with 3 linear-attention blocks then 1 gated-attention block per stage",
    context_length="262,144 native",
    base_weight_size_gib="~51.8 GiB safetensors / ~53.8 GB F16 GGUF",
    releases=(
        ReleaseArtifact(
            slug="Qwen3.6-27B-TQ1_0",
            weight_profile="TQ1_0",
            profile_type="uniform",
            target_size_gib="~5.5-6.5 GiB",
            quality_band="research floor only",
            mac_target="32 GB+",
            runtimes=("llama.cpp", "opentq-metal"),
            priority=7,
            notes="Minimum-footprint stress release. Useful as a lower bound, not a first public recommendation.",
        ),
        ReleaseArtifact(
            slug="Qwen3.6-27B-TQ2_0",
            weight_profile="TQ2_0",
            profile_type="uniform",
            target_size_gib="~8.0-9.5 GiB",
            quality_band="aggressive compact",
            mac_target="32 GB",
            runtimes=("llama.cpp", "opentq-metal"),
            priority=5,
            notes="Compact long-context option where RAM headroom matters more than maximal fidelity.",
        ),
        ReleaseArtifact(
            slug="Qwen3.6-27B-TQ3_SB4",
            weight_profile="TQ3_SB4",
            profile_type="uniform",
            target_size_gib="~12.6-13.3 GiB",
            quality_band="compact daily driver",
            mac_target="32 GB",
            runtimes=("llama.cpp", "opentq-metal"),
            priority=2,
            notes="Closest open counterpart to the public 3.5-bit WHT family. Expected to compete directly with strong 3-4 bit GGUFs.",
        ),
        ReleaseArtifact(
            slug="Qwen3.6-27B-TQ4_SB4",
            weight_profile="TQ4_SB4",
            profile_type="uniform",
            target_size_gib="~15.5-16.8 GiB",
            quality_band="balanced default",
            mac_target="32 GB / 48 GB",
            runtimes=("llama.cpp", "opentq-metal"),
            priority=1,
            notes="Primary first-wave release. Best expected blend of quality, simplicity, and deployability.",
        ),
        ReleaseArtifact(
            slug="Qwen3.6-27B-TQ4R2",
            weight_profile="TQ4R2",
            profile_type="uniform-residual",
            target_size_gib="~18.5-20.5 GiB",
            quality_band="quality-first",
            mac_target="48 GB preferred, 32 GB possible with tighter ctx",
            runtimes=("llama.cpp", "opentq-metal"),
            priority=3,
            notes="4+2 residual path meant to beat plain 6-bit class formats on quality-per-byte.",
        ),
        ReleaseArtifact(
            slug="Qwen3.6-27B-TQ4R4",
            weight_profile="TQ4R4",
            profile_type="uniform-residual",
            target_size_gib="~24.0-28.0 GiB",
            quality_band="near-lossless reference",
            mac_target="48 GB+",
            runtimes=("llama.cpp", "opentq-metal"),
            priority=6,
            notes="Reference-grade release for regression testing and high-end Macs.",
        ),
        ReleaseArtifact(
            slug="Qwen3.6-27B-TQ_BAL_DENSE",
            weight_profile="mixed: embed/lm_head=TQ4R4, self_attn=TQ4R2, linear_attn=TQ4_SB4, mlp=TQ3_SB4, visual=TQ4_SB4, norms+state=bf16",
            profile_type="mixed",
            target_size_gib="~14.5-16.0 GiB",
            quality_band="flagship mixed profile",
            mac_target="32 GB / 48 GB",
            runtimes=("llama.cpp", "opentq-metal"),
            priority=4,
            notes="Dense-hybrid flagship release tailored to Qwen3.6-27B instead of using one global bit policy everywhere.",
        ),
    ),
    phases=(
        WorkPhase(
            name="P0 inventory",
            goal="Freeze tensor inventory, role classification, and calibration corpus.",
            deliverables=(
                "HF safetensors index snapshot",
                "tensor-role classifier",
                "vision-tower role split",
                "calibration set manifest for long-context coding prompts",
            ),
            exit_criteria=(
                "All tensors classified into stable roles",
                "Role counts checked against model architecture",
                "Vision and language towers are handled explicitly in the recipe",
            ),
        ),
        WorkPhase(
            name="P1 uniform 4-bit path",
            goal="Ship the first end-to-end quantization path with the lowest implementation risk.",
            deliverables=(
                "TQ4_SB4 quantizer path",
                "GGUF emitter draft",
                "perplexity and wall-clock benchmarks",
            ),
            exit_criteria=(
                "Weights quantize end-to-end for Qwen3.6-27B",
                "llama.cpp loader branch can run the artifact",
            ),
        ),
        WorkPhase(
            name="P2 compact baseline",
            goal="Derive the compact public release that should fit 32 GB Macs comfortably.",
            deliverables=(
                "TQ3_SB4 artifact",
                "comparison against Q3_K_M / public TQ3_4S",
            ),
            exit_criteria=(
                "TQ3_SB4 quality gap is characterized",
                "32 GB memory target is validated at realistic context",
            ),
        ),
        WorkPhase(
            name="P3 residual line",
            goal="Bring up TQ4R2 then TQ4R4 for better quality-per-byte.",
            deliverables=(
                "TQ4R2 artifact",
                "TQ4R4 artifact",
                "residual dequant path in runtime",
            ),
            exit_criteria=(
                "Residual formats beat the non-residual baseline on evals",
                "runtime path is stable under long prompts",
            ),
        ),
        WorkPhase(
            name="P4 flagship mixed profile",
            goal="Exploit Qwen3.6-27B's hybrid layout with tensor-role-aware bit allocation.",
            deliverables=(
                "TQ_BAL_DENSE policy",
                "mixed manifest format",
                "publishable flagship checkpoint",
            ),
            exit_criteria=(
                "Mixed profile outperforms uniform releases at similar size",
                "Artifact is simple enough to document and reproduce",
            ),
        ),
        WorkPhase(
            name="P5 runtime upgrades",
            goal="Layer runtime-specific gains on top of the weight family.",
            deliverables=(
                "llama.cpp patchset for OpenTQ types",
                "opentq-metal int4 compressed-domain KV path",
                "DFlash adapter plan for Qwen3.6-27B",
            ),
            exit_criteria=(
                "Weight and KV experiments are benchmarked separately",
                "Speculative path has a credible integration route",
            ),
        ),
    ),
)


RECIPES: dict[str, ModelRecipe] = {
    QWEN36_27B_RECIPE.key: QWEN36_27B_RECIPE,
}


def get_recipe(key: str) -> ModelRecipe:
    normalized = key.lower()
    try:
        return RECIPES[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(RECIPES))
        raise KeyError(f"unknown recipe {key!r}; available: {available}") from exc


def recipe_to_dict(recipe: ModelRecipe) -> dict[str, object]:
    return {
        "key": recipe.key,
        "model_id": recipe.model_id,
        "family": recipe.family,
        "architecture": recipe.architecture,
        "context_length": recipe.context_length,
        "base_weight_size_gib": recipe.base_weight_size_gib,
        "releases": [asdict(release) for release in recipe.releases],
        "phases": [asdict(phase) for phase in recipe.phases],
    }


def recipe_markdown(recipe: ModelRecipe) -> str:
    lines = [
        f"# {recipe.model_id}",
        "",
        f"- Family: `{recipe.family}`",
        f"- Architecture: {recipe.architecture}",
        f"- Context length: {recipe.context_length}",
        f"- Base weight size: {recipe.base_weight_size_gib}",
        "",
        "## Release Matrix",
        "",
        "| Priority | Release | Profile | Target size | Quality | Mac target | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for release in sorted(recipe.releases, key=lambda item: item.priority):
        lines.append(
            f"| {release.priority} | `{release.slug}` | `{release.weight_profile}` | {release.target_size_gib} | {release.quality_band} | {release.mac_target} | {release.notes} |"
        )

    lines.extend(
        [
            "",
            "## Phases",
            "",
        ]
    )
    for phase in recipe.phases:
        lines.append(f"### {phase.name}")
        lines.append(phase.goal)
        lines.append("")
        lines.append("Deliverables:")
        for item in phase.deliverables:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Exit criteria:")
        for item in phase.exit_criteria:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
