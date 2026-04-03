from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import optuna
except ImportError as exc:  # pragma: no cover - depends on local environment
    optuna = None
    _OPTUNA_IMPORT_ERROR = exc
else:
    _OPTUNA_IMPORT_ERROR = None


SearchSpace = Mapping[str, Tuple[str, Any]]
ObjectiveFn = Callable[[Any, Dict[str, Any], "TrialReporter"], float]


@dataclass
class TrialReporter:
    """Thin wrapper used by training code to report intermediate metrics."""

    trial: Any
    metric_name: str = "valid_mrr"
    history: List[Tuple[int, float]] = field(default_factory=list)

    def report(self, step: int, value: float) -> None:
        value = float(value)
        step = int(step)
        self.history.append((step, value))
        self.trial.report(value, step)

    def should_prune(self) -> bool:
        return bool(self.trial.should_prune())

    def report_and_prune(self, step: int, value: float) -> None:
        self.report(step, value)
        if self.should_prune():
            raise optuna.TrialPruned()


class OptunaTPEHyperbandHPO:
    """
    Standalone mature HPO runner.

    This file is intentionally not wired into the current training entrypoints.
    It is designed so later integration can replace the custom RF+EI pipeline with:
    - Optuna TPESampler for configuration proposal
    - Optuna HyperbandPruner for multi-fidelity pruning
    """

    def __init__(
        self,
        search_space: SearchSpace,
        *,
        study_name: str = "redgnn-optuna",
        direction: str = "maximize",
        seed: int = 1234,
        metric_name: str = "valid_mrr",
        n_startup_trials: int = 10,
        n_ei_candidates: int = 64,
        multivariate: bool = True,
        group: bool = True,
        min_resource: int = 1,
        max_resource: Any = "auto",
        reduction_factor: int = 3,
    ) -> None:
        self._ensure_optuna()
        self.search_space = dict(search_space)
        self.study_name = study_name
        self.direction = direction
        self.seed = int(seed)
        self.metric_name = metric_name
        self.n_startup_trials = int(n_startup_trials)
        self.n_ei_candidates = int(n_ei_candidates)
        self.multivariate = bool(multivariate)
        self.group = bool(group)
        self.min_resource = int(min_resource)
        self.max_resource = max_resource
        self.reduction_factor = int(reduction_factor)
        self.study = None

    @staticmethod
    def _ensure_optuna() -> None:
        if optuna is None:
            raise ImportError(
                "Optuna is not installed. Install it first, for example: pip install optuna"
            ) from _OPTUNA_IMPORT_ERROR

    def _build_sampler(self) -> Any:
        return optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=self.n_startup_trials,
            n_ei_candidates=self.n_ei_candidates,
            multivariate=self.multivariate,
            group=self.group,
        )

    def _build_pruner(self) -> Any:
        return optuna.pruners.HyperbandPruner(
            min_resource=self.min_resource,
            max_resource=self.max_resource,
            reduction_factor=self.reduction_factor,
        )

    def create_study(self) -> Any:
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=self._build_sampler(),
            pruner=self._build_pruner(),
        )
        return self.study

    def _coerce_start_config(self, config: Mapping[str, Any]) -> Dict[str, Any]:
        fixed: Dict[str, Any] = {}
        for name, spec in self.search_space.items():
            if name not in config:
                continue

            hp_type, hp_range = spec
            value = config[name]

            if hp_type == "choice":
                if value not in hp_range:
                    raise ValueError(f'start_config["{name}"]={value} not in {list(hp_range)}')
                fixed[name] = value
            elif hp_type == "uniform":
                lo, hi = float(hp_range[0]), float(hp_range[1])
                value = float(value)
                if value < lo or value > hi:
                    raise ValueError(f'start_config["{name}"]={value} out of range [{lo}, {hi}]')
                fixed[name] = value
            elif hp_type == "loguniform":
                lo, hi = float(hp_range[0]), float(hp_range[1])
                value = float(value)
                if value < lo or value > hi:
                    raise ValueError(f'start_config["{name}"]={value} out of range [{lo}, {hi}]')
                fixed[name] = value
            elif hp_type == "int":
                lo, hi = int(hp_range[0]), int(hp_range[1])
                step = int(hp_range[2]) if len(hp_range) >= 3 else 1
                value = int(value)
                if value < lo or value > hi or ((value - lo) % step != 0):
                    raise ValueError(
                        f'start_config["{name}"]={value} out of int range [{lo}, {hi}] with step={step}'
                    )
                fixed[name] = value
            elif hp_type == "int_loguniform":
                lo, hi = int(hp_range[0]), int(hp_range[1])
                value = int(value)
                if value < lo or value > hi:
                    raise ValueError(f'start_config["{name}"]={value} out of range [{lo}, {hi}]')
                fixed[name] = value
            else:
                raise ValueError(f"Unsupported search-space type: {hp_type}")
        return fixed

    def enqueue_trials(self, start_configs: Iterable[Mapping[str, Any]]) -> None:
        if self.study is None:
            self.create_study()
        for config in start_configs:
            self.study.enqueue_trial(self._coerce_start_config(config))

    def suggest_config(self, trial: Any) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for name, spec in self.search_space.items():
            hp_type, hp_range = spec

            if hp_type == "choice":
                config[name] = trial.suggest_categorical(name, list(hp_range))
            elif hp_type == "uniform":
                config[name] = trial.suggest_float(name, float(hp_range[0]), float(hp_range[1]))
            elif hp_type == "loguniform":
                config[name] = trial.suggest_float(
                    name, float(hp_range[0]), float(hp_range[1]), log=True
                )
            elif hp_type == "int":
                lo, hi = int(hp_range[0]), int(hp_range[1])
                step = int(hp_range[2]) if len(hp_range) >= 3 else 1
                config[name] = trial.suggest_int(name, lo, hi, step=step)
            elif hp_type == "int_loguniform":
                lo, hi = int(hp_range[0]), int(hp_range[1])
                config[name] = trial.suggest_int(name, lo, hi, log=True)
            else:
                raise ValueError(f"Unsupported search-space type: {hp_type}")
        return config

    def optimize(
        self,
        objective_fn: ObjectiveFn,
        *,
        n_trials: int,
        timeout: Optional[float] = None,
        start_configs: Optional[Iterable[Mapping[str, Any]]] = None,
        catch: Sequence[type[BaseException]] = (),
    ) -> Any:
        if self.study is None:
            self.create_study()
        if start_configs is not None:
            self.enqueue_trials(start_configs)

        def _objective(trial: Any) -> float:
            config = self.suggest_config(trial)
            reporter = TrialReporter(trial=trial, metric_name=self.metric_name)
            value = float(objective_fn(trial, config, reporter))
            trial.set_user_attr("config", config)
            trial.set_user_attr("metric_name", self.metric_name)
            trial.set_user_attr("history", reporter.history)
            return value

        self.study.optimize(
            _objective,
            n_trials=int(n_trials),
            timeout=timeout,
            catch=tuple(catch),
            gc_after_trial=True,
        )
        return self.study

    @property
    def best_value(self) -> float:
        if self.study is None or self.study.best_trial is None:
            raise RuntimeError("No completed study is available.")
        return float(self.study.best_value)

    @property
    def best_config(self) -> Dict[str, Any]:
        if self.study is None or self.study.best_trial is None:
            raise RuntimeError("No completed study is available.")
        return dict(self.study.best_trial.user_attrs.get("config", self.study.best_params))

    def top_trials(self, k: int = 10) -> List[Dict[str, Any]]:
        if self.study is None:
            return []

        reverse = self.direction == "maximize"
        complete = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        complete.sort(key=lambda t: t.value, reverse=reverse)

        results = []
        for trial in complete[: int(k)]:
            results.append(
                {
                    "number": trial.number,
                    "value": float(trial.value),
                    "params": dict(trial.params),
                    "history": list(trial.user_attrs.get("history", [])),
                }
            )
        return results
