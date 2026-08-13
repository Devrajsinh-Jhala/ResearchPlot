"""ResearchPlot 2 project orchestration built on immutable specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .compliance import Policy, Report, RuleEngine
from .models import OutputFormat, RuleLevel, RulePhase, VenueProfile, VenueRule
from .observations import Observation, ObservationSet
from .planning import (
    CompliancePlan,
    CoverageRequirement,
    ExportPlan,
    FigurePlan,
    PlanAssessment,
    PlanEvidence,
    plan_export,
)
from .profile_lock import load_profile_lock, verify_profile_lock
from .registry import resolve_profile
from .specs import FigureSpec, ManuscriptFormat, ProjectSpec, Waiver
from .style import StyleContext
from .submission import BundleResult, Submission
from .target import Target, coerce_format
from .target import target as make_target

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .manuscript import ManuscriptAudit
    from .transactional_export import ExportResult


def _rule_matches(rule: VenueRule, target: Target, output_format: OutputFormat | None) -> bool:
    return rule.applies_to.matches(
        role=target.role,
        content_kind=target.content,
        output_format=output_format,
        width=target.width,
    )


def _bundle_report(figure: FigureSpec, target: Target) -> Report:
    alt_text = (figure.alt_text or "").strip().casefold()
    caption = (figure.caption or "").strip().casefold()
    source_paths = [*figure.source_data]
    if figure.data_table is not None:
        source_paths.append(figure.data_table)
    for panel in figure.panels:
        source_paths.extend(panel.source_data)
    source_data_present = bool(source_paths) and all(path.is_file() for path in source_paths)
    attachments_present = bool(figure.attachments) and all(
        path.is_file() for path in figure.attachments
    )
    panel_descriptions_complete = bool(figure.panels) and all(
        panel.description is not None
        or panel.alt_text is not None
        or panel.long_description is not None
        for panel in figure.panels
    )
    panel_order_complete = bool(figure.panels) and all(
        panel.order is not None for panel in figure.panels
    )
    observations = ObservationSet(
        (
            Observation("metadata.alt_text.present", bool(alt_text), phase="bundle"),
            Observation("metadata.caption.present", bool(caption), phase="bundle"),
            Observation("metadata.source_data.present", source_data_present, phase="bundle"),
            Observation(
                "metadata.long_description.present",
                bool(figure.long_description),
                phase="bundle",
            ),
            Observation("metadata.key_trends.present", bool(figure.key_trends), phase="bundle"),
            Observation(
                "metadata.data_table.present", figure.data_table is not None, phase="bundle"
            ),
            Observation("metadata.attachments.present", attachments_present, phase="bundle"),
            Observation(
                "metadata.panel_descriptions.complete",
                panel_descriptions_complete,
                phase="bundle",
            ),
            Observation(
                "metadata.panel_order.complete",
                panel_order_complete,
                phase="bundle",
            ),
            Observation(
                "metadata.figure_number.present", figure.number is not None, phase="bundle"
            ),
            Observation(
                "metadata.alt_text.distinct_from_caption",
                bool(alt_text) and alt_text != caption,
                phase="bundle",
            ),
        )
    )
    return RuleEngine().evaluate(
        target.profile,
        observations,
        target.context(),
        phase=RulePhase.BUNDLE.value,
        attestations=figure.attestations,
    )


def _coverage_requirements(figure: FigureSpec, target: Target) -> tuple[CoverageRequirement, ...]:
    """Compile required profile rules into artifact-aware evidence expectations."""

    requirements: dict[tuple[str, str | None, str], CoverageRequirement] = {}
    required_deliverables = tuple(item for item in figure.deliverables if item.required)
    for rule in target.profile.rules:
        if rule.level is not RuleLevel.REQUIRED:
            continue

        if RulePhase.BUNDLE in rule.phases and _rule_matches(rule, target, None):
            item = CoverageRequirement(
                figure_id=figure.id,
                rule_id=rule.id,
                phases=(RulePhase.BUNDLE,),
            )
            requirements[item.key] = item

        file_requirements = 0
        if RulePhase.FILE in rule.phases:
            for deliverable in required_deliverables:
                output_format = cast(OutputFormat, deliverable.format)
                if not _rule_matches(rule, target, output_format):
                    continue
                item = CoverageRequirement(
                    figure_id=figure.id,
                    deliverable_id=deliverable.id,
                    rule_id=rule.id,
                    phases=(RulePhase.FILE,),
                )
                requirements[item.key] = item
                file_requirements += 1

        # A live observation is required only when no committed required artifact can
        # establish the same rule. This avoids counting LIVE and FILE as two separate
        # obligations while ensuring live-only typography rules cannot disappear.
        if (
            RulePhase.LIVE in rule.phases
            and file_requirements == 0
            and _rule_matches(rule, target, None)
        ):
            item = CoverageRequirement(
                figure_id=figure.id,
                rule_id=rule.id,
                phases=(RulePhase.LIVE,),
            )
            requirements[item.key] = item
    return tuple(requirements.values())


@dataclass(frozen=True, slots=True)
class Project:
    """Resolved schema-v3 project with planning and coverage-aware checks."""

    spec: ProjectSpec
    profile: VenueProfile = field(init=False)
    _plan: CompliancePlan = field(init=False, repr=False)

    def __post_init__(self) -> None:
        profile = resolve_profile(self.spec.profile)
        object.__setattr__(self, "profile", profile)
        figure_plans: list[FigurePlan] = []
        requirements: list[CoverageRequirement] = []
        for figure in self.spec.figures:
            for rule_id in figure.attestations:
                rule = profile.get_rule(rule_id)
                if rule is None:
                    raise ValueError(
                        f"Attestation {figure.id}/{rule_id} references an unknown profile rule."
                    )
                if rule.verification.value != "manual":
                    raise ValueError(
                        f"Attestation {figure.id}/{rule_id} targets a rule that is not manual."
                    )
            for rule_id, waiver in figure.waivers.items():
                if profile.get_rule(rule_id) is None:
                    raise ValueError(
                        f"Waiver {figure.id}/{rule_id} references an unknown profile rule."
                    )
                if isinstance(waiver, Waiver) and waiver.profile_digest != profile.digest:
                    raise ValueError(
                        f"Waiver {figure.id}/{rule_id} is bound to a different profile digest."
                    )
            target = make_target(
                profile,
                role=figure.role,
                width=figure.width,
                content=figure.content,
            )
            formats = tuple(cast(OutputFormat, item.format) for item in figure.deliverables)
            preferred = next(
                (cast(OutputFormat, item.format) for item in figure.deliverables if item.preferred),
                None,
            )
            export = plan_export(target, formats=formats, preferred=preferred)
            figure_plans.append(
                FigurePlan(
                    figure.id,
                    target,
                    export,
                    tuple(item.id for item in figure.deliverables),
                    tuple(sorted(figure.active_waiver_rule_ids)),
                )
            )
            requirements.extend(_coverage_requirements(figure, target))
        capability_gaps: tuple[str, ...] = ()
        if self.spec.manuscript is not None and self.spec.manuscript.required:
            capability_gaps = (
                "Required compiled-manuscript placement matching needs separate manuscript "
                "evidence; figure-only checks cannot establish it.",
            )
        object.__setattr__(
            self,
            "_plan",
            CompliancePlan(
                profile,
                tuple(figure_plans),
                tuple(requirements),
                capability_gaps,
            ),
        )

    @classmethod
    def load(cls, path: str | Path = "researchplot.toml") -> Project:
        """Load a strict schema-v3 TOML project and resolve its pinned profile."""

        return cls(ProjectSpec.load(path))

    @property
    def compliance_plan(self) -> CompliancePlan:
        return self._plan

    def plan(self, *, frozen: bool = False) -> ExecutablePlan:
        """Return the executable compliance plan used by the primary v2 workflow.

        ``frozen=True`` verifies the configured profile lock immediately and again
        before checking any artifacts.
        """

        result = ExecutablePlan(self, frozen=frozen)
        if frozen:
            result.verify_lock()
        return result

    def figure(self, figure_id: str) -> FigureTarget:
        """Return a figure-scoped style/check/export facade."""

        spec = self.spec.figure(figure_id)
        plan = next(item for item in self._plan.figures if item.figure_id == figure_id)
        return FigureTarget(self, spec, plan.target, plan.export)

    def target(self, figure_id: str) -> Target:
        """Return the resolved target used by one figure plan."""

        try:
            return next(item.target for item in self._plan.figures if item.figure_id == figure_id)
        except StopIteration as exc:
            choices = ", ".join(item.figure_id for item in self._plan.figures)
            raise KeyError(f"Unknown figure {figure_id!r}. Available figures: {choices}.") from exc

    def assess(self, *evidence: PlanEvidence) -> PlanAssessment:
        """Assess caller-supplied live, file, or bundle evidence against the plan."""

        return self._plan.assess(evidence)

    def audit_manuscript(self, *, max_pages: int = 2_000) -> ManuscriptAudit:
        """Audit the configured compiled PDF and conservatively match figure placements.

        Placement matching is coverage evidence, not an acceptance guarantee. Missing or
        ambiguous placements remain unresolved and are never inferred as passing.
        """

        manuscript = self.spec.manuscript
        if manuscript is None:
            raise ValueError("The project does not configure a manuscript.")
        if cast(ManuscriptFormat, manuscript.format) is not ManuscriptFormat.PDF:
            raise ValueError("Compiled manuscript auditing currently supports PDF only.")
        if not manuscript.path.is_file():
            raise FileNotFoundError(f"Configured manuscript not found: {manuscript.path}")
        from .manuscript import audit_manuscript_pdf

        return audit_manuscript_pdf(
            manuscript.path,
            max_pages=max_pages,
            figures=self.spec.figures,
            matching_hints=manuscript.matching_hints,
        )

    def _figure_evidence(
        self,
        figure: FigureSpec,
        *,
        live: Figure | None = None,
        include_artifacts: bool = True,
    ) -> tuple[PlanEvidence, ...]:
        target = self.target(figure.id)
        evidence: list[PlanEvidence] = []
        if live is not None:
            evidence.append(
                PlanEvidence(
                    figure.id,
                    target.validate(live, attestations=dict(figure.attestation_statements)),
                )
            )
        bundle = _bundle_report(figure, target)
        if bundle.findings:
            evidence.append(PlanEvidence(figure.id, bundle))
        if include_artifacts:
            for deliverable in figure.deliverables:
                if deliverable.path is None or not deliverable.path.is_file():
                    continue
                report = target.audit(
                    deliverable.path,
                    attestations=dict(figure.attestation_statements),
                )
                evidence.append(PlanEvidence(figure.id, report, deliverable.id))
        return tuple(evidence)

    def check(
        self,
        *,
        live_figures: Mapping[str, Figure] | None = None,
    ) -> PlanAssessment:
        """Audit configured artifacts and bundle metadata without hiding live-only gaps."""

        evidence: list[PlanEvidence] = []
        live_figures = live_figures or {}
        unknown_live = set(live_figures) - {item.id for item in self.spec.figures}
        if unknown_live:
            raise ValueError(
                "Live figures reference unknown ids: " + ", ".join(sorted(unknown_live)) + "."
            )
        for figure in self.spec.figures:
            evidence.extend(
                self._figure_evidence(
                    figure,
                    live=live_figures.get(figure.id),
                )
            )
        return self._plan.assess(evidence)

    def bundle(
        self,
        output_dir: str | Path,
        *,
        live_figures: Mapping[str, Figure] | None = None,
    ) -> BundleResult:
        """Build a v1-compatible staged bundle from this immutable project.

        A live figure can produce every planned representation. Existing-file projects
        must currently select one preferred representation per logical figure; generic
        attachments and multi-file source-data sets require a future bundle manifest.
        """

        provided = dict(live_figures or {})
        unknown = set(provided) - {item.id for item in self.spec.figures}
        if unknown:
            raise ValueError("Live figures reference unknown ids: " + ", ".join(sorted(unknown)))
        submission = Submission(
            self.profile,
            output_dir=output_dir,
            policy=self.spec.policy,
        )
        for figure in self.spec.figures:
            if figure.attachments:
                raise ValueError(
                    f"Figure {figure.id!r} has generic attachments; Project.bundle cannot "
                    "yet represent them without losing manifest semantics."
                )
            sources = [*figure.source_data]
            if figure.data_table is not None:
                sources.append(figure.data_table)
            for panel in figure.panels:
                sources.extend(panel.source_data)
            source_keys = tuple(dict.fromkeys(path.resolve() for path in sources))
            if len(source_keys) > 1:
                raise ValueError(
                    f"Figure {figure.id!r} has multiple source-data files; Project.bundle "
                    "currently accepts one per figure."
                )

            live = provided.get(figure.id)
            formats: tuple[OutputFormat | str, ...] | None = None
            name = figure.id
            asset: Figure | Path
            if live is not None:
                asset = live
                formats = self.figure(figure.id).export_plan.selected_formats
            else:
                candidates = [
                    item
                    for item in figure.deliverables
                    if item.required and item.path is not None and item.path.is_file()
                ]
                preferred = next((item for item in candidates if item.preferred), None)
                if preferred is None and len(candidates) == 1:
                    preferred = candidates[0]
                if preferred is None:
                    raise FileNotFoundError(
                        f"Figure {figure.id!r} needs a live figure or one existing preferred "
                        "deliverable for bundling."
                    )
                asset = cast(Path, preferred.path)
                name = asset.name
            submission.add(
                name,
                asset,
                role=figure.role,
                width=figure.width,
                content=figure.content,
                formats=formats,
                alt_text=figure.alt_text,
                caption=figure.caption,
                source_data=source_keys[0] if source_keys else None,
                attestations=dict(figure.attestation_statements),
            )
        return submission.build()

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec.to_dict(),
            "profile": self.profile.to_dict(),
            "compliance_plan": self._plan.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ExecutablePlan:
    """Thin executable facade over an immutable :class:`CompliancePlan`."""

    project: Project = field(repr=False)
    frozen: bool = False

    @property
    def compliance_plan(self) -> CompliancePlan:
        return self.project.compliance_plan

    def verify_lock(self) -> None:
        """Verify exact profile and evidence digests without inspecting artifacts."""

        lock_path = self.project.spec.lock_path
        if lock_path is None:
            raise ValueError(
                "Frozen planning requires project.lock to name a committed profile lock."
            )
        lock = load_profile_lock(lock_path)
        verify_profile_lock(lock, self.project.profile)

    def check(
        self,
        *,
        live_figures: Mapping[str, Figure] | None = None,
    ) -> PlanAssessment:
        if self.frozen:
            self.verify_lock()
        return self.project.check(live_figures=live_figures)

    def to_dict(self) -> dict[str, object]:
        return {
            "frozen": self.frozen,
            "lock": self.project.spec.lock_path.as_posix()
            if self.project.spec.lock_path is not None
            else None,
            "compliance_plan": self.compliance_plan.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FigureTarget:
    """Figure-scoped style, check, audit, and export conveniences."""

    project: Project = field(repr=False)
    spec: FigureSpec
    target: Target
    export_plan: ExportPlan

    @property
    def id(self) -> str:
        return self.spec.id

    def style(
        self,
        *,
        deliverable: str | None = None,
        latex: bool = False,
        overrides: Mapping[str, Any] | None = None,
    ) -> StyleContext:
        """Create the reversible Matplotlib context resolved for this figure."""

        if deliverable is not None:
            choices = {item.id for item in self.spec.deliverables}
            if deliverable not in choices:
                raise KeyError(
                    f"Unknown deliverable {deliverable!r} for figure {self.id!r}; "
                    f"choose from: {', '.join(sorted(choices))}."
                )
        return self.target.style(latex=latex, overrides=dict(overrides or {}))

    def validate(self, fig: Figure) -> Report:
        """Return the raw live-phase report for callers that do not need coverage."""

        return self.target.validate(fig, attestations=dict(self.spec.attestation_statements))

    def audit(self, path: str | Path) -> Report:
        """Return the raw file-phase report for an existing representation."""

        return self.target.audit(path, attestations=dict(self.spec.attestation_statements))

    def check(
        self,
        *,
        fig: Figure | None = None,
        include_artifacts: bool = True,
    ) -> PlanAssessment:
        """Assess this figure without unrelated project figures becoming coverage gaps."""

        evidence = self.project._figure_evidence(
            self.spec,
            live=fig,
            include_artifacts=include_artifacts,
        )
        figure_plan = next(
            item for item in self.project.compliance_plan.figures if item.figure_id == self.id
        )
        requirements = tuple(
            item for item in self.project.compliance_plan.requirements if item.figure_id == self.id
        )
        plan = CompliancePlan(
            self.project.profile,
            (figure_plan,),
            requirements,
            self.project.compliance_plan.capability_gaps,
        )
        return plan.assess(evidence)

    def export(
        self,
        fig: Figure,
        target_path: str | Path | None = None,
        *,
        formats: Sequence[OutputFormat | str] | None = None,
        policy: Policy | str | None = None,
        dpi: int | None = None,
        overwrite: bool = False,
        metadata: Mapping[str, object] | None = None,
        **savefig_kwargs: Any,
    ) -> ExportResult:
        """Export through the resolved plan, defaulting to declared artifact paths."""

        selected_formats = (
            tuple(coerce_format(item) for item in formats)
            if formats is not None
            else self.export_plan.selected_formats
        )
        selected_path: Path
        export_formats: tuple[OutputFormat | str, ...] | None
        if target_path is None:
            candidates = [
                item
                for item in self.spec.deliverables
                if item.path is not None and cast(OutputFormat, item.format) in selected_formats
            ]
            if len(candidates) != len(selected_formats):
                raise ValueError(
                    f"Figure {self.id!r} needs target_path because not every selected "
                    "deliverable declares a path."
                )
            paths = [cast(Path, item.path) for item in candidates]
            stems = {path.with_suffix("") for path in paths}
            if len(paths) == 1:
                selected_path = paths[0]
                export_formats = None
            elif len(stems) == 1:
                selected_path = next(iter(stems))
                export_formats = selected_formats
            else:
                raise ValueError(
                    f"Figure {self.id!r} declares different output stems; pass target_path "
                    "and formats explicitly."
                )
        else:
            selected_path = Path(target_path)
            export_formats = None if selected_path.suffix and formats is None else selected_formats

        manifest_metadata: dict[str, object] = {
            "figure_id": self.id,
            "number": self.spec.number,
            "caption": self.spec.caption,
            "alt_text": self.spec.alt_text,
            "long_description": self.spec.long_description,
            "key_trends": list(self.spec.key_trends),
            "panels": [
                panel.to_dict(relative_to=self.project.spec.root) for panel in self.spec.panels
            ],
            "attestations": self.spec.to_dict().get("attestations", {}),
            "waivers": self.spec.to_dict().get("waivers", {}),
        }
        manifest_metadata.update(metadata or {})
        return self.target.export(
            fig,
            selected_path,
            formats=export_formats,
            policy=policy if policy is not None else self.project.spec.policy,
            dpi=dpi,
            overwrite=overwrite,
            attestations=dict(self.spec.attestation_statements),
            metadata=manifest_metadata,
            **savefig_kwargs,
        )


class PlanPolicyError(ValueError):
    """Raised when a coverage-aware project assessment is blocked by policy."""

    def __init__(self, assessment: PlanAssessment, policy: Policy | str) -> None:
        self.assessment = assessment
        self.policy = Policy(policy)
        super().__init__(
            f"{self.policy.value} policy blocked {assessment.profile}: {assessment.verdict.value}."
        )


def enforce_assessment(
    assessment: PlanAssessment, policy: Policy | str = Policy.COMPLETE
) -> PlanAssessment:
    """Return an assessment or raise when the selected policy blocks it."""

    selected = Policy(policy)
    if assessment.blocks(selected):
        raise PlanPolicyError(assessment, selected)
    return assessment
