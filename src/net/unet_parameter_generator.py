from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, Any, List, Optional, Callable
import math
import optuna
from itertools import combinations

EncoderSpec = Tuple[int, int]
TrialCallback = Callable[
    [optuna.Study, optuna.trial.FrozenTrial, int, int, List[EncoderSpec], Optional[float], bool, Optional[str]], None]


@dataclass
class UNetParamGenerator:
    net_input_sizes: Sequence[int]
    num_layer_channels: Sequence[int]
    num_layers: Sequence[int]
    filter_sizes: Sequence[int]
    in_channels: int = 3
    num_classes: int = 1
    evaluator: Any = None

    def generate(
            self,
            *,
            n_trials_per_setting: int = 20,
            sampler: Optional[optuna.samplers.BaseSampler] = None,
            pruner: Optional[optuna.pruners.BasePruner] = None,
            study_name_prefix: str = "unet_param_search",
            verbose: bool = False,
            trial_callback: Optional[TrialCallback] = None,
            storage_url: Optional[str] = None,
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        if self.evaluator is None:
            raise ValueError("evaluator must not be None (has to provide .evaluate(...)).")

        results = {}
        for L in self.num_layers:
            for S in self.net_input_sizes:
                study_name = f"{study_name_prefix}_L{L}_S{S}"
                if verbose:
                    print(f"[Gen] Start study: {study_name}  (trials={n_trials_per_setting})")

                if not self._is_input_divisible(S, L):
                    if verbose:
                        print(f"[Gen] Skip: input_size={S} not divisible by 2**(L-1) with L={L}")
                    results[(L, S)] = {"best_params": None, "best_value": None, "study": None}
                    continue

                # ---- NEU: Sequenzen einmal bauen
                ch_sequences = self._get_channel_sequences(L)
                if not ch_sequences:
                    if verbose:
                        print(f"[Gen] Skip: no strictly increasing channel sequences for L={L}")
                    results[(L, S)] = {"best_params": None, "best_value": None, "study": None}
                    continue

                study = optuna.create_study(
                    direction="maximize",
                    sampler=sampler if sampler is not None else optuna.samplers.TPESampler(),
                    pruner=pruner if pruner is not None else optuna.pruners.MedianPruner(n_startup_trials=5),
                    study_name=study_name,
                    storage=storage_url,
                    load_if_exists=bool(storage_url),
                )

                def objective(trial: optuna.trial.Trial) -> float:
                    idx = trial.suggest_int(f"ch_seq_idx_L{L}", 0, len(ch_sequences) - 1)
                    chs = ch_sequences[idx]  # tuple mit Länge L

                    specs: list[tuple[int, int]] = []
                    for i in range(L):
                        ks = trial.suggest_categorical(f"layer_{i}_ks", list(self.filter_sizes))
                        ks = int(ks)
                        if ks % 2 == 0 or ks < 3:
                            raise optuna.exceptions.TrialPruned(f"invalid kernel_size={ks}")
                        specs.append((ks, int(chs[i])))

                    ok, reason = self.perform_param_sanity_check(S, specs)
                    if not ok:
                        raise optuna.exceptions.TrialPruned(f"sanity check failed: {reason}")

                    val = float(self.evaluate_model(S, specs))

                    complexity = 0.0
                    for ks, ch in specs:
                        complexity += (ks * ks) * (ch * ch)

                    ref = float(L * (3 * 3) * (32 * 32))  # = L * 9 * 1024
                    pen = 0.0
                    if ref > 0:
                        ratio = max(1e-9, complexity / ref)
                        pen = max(0.0, math.log10(ratio))

                    alpha = getattr(self, "alpha_penalty", 0.08)  # anpassbar
                    score = float(val - alpha * pen)

                    trial.set_user_attr("val_score", val)
                    trial.set_user_attr("complexity", complexity)
                    trial.set_user_attr("penalty", pen)
                    trial.set_user_attr("specs", specs)
                    print(f"[Objective] L={L} S={S} specs={specs} | val={val:.4f} "
                          f"complexity={complexity:.1f} pen={pen:.3f} -> score={score:.4f}")

                    if trial_callback:
                        trial_callback(study, trial, L, S, specs, float(score), False, None)
                    return score

                study.optimize(objective, n_trials=n_trials_per_setting, show_progress_bar=False)

                from optuna.trial import TrialState
                completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
                if not completed:
                    if verbose:
                        print(f"[Gen] No completed trials for {study_name} (all pruned/failed).")
                    results[(L, S)] = {"best_params": None, "best_value": None, "study": study}
                    continue

                best_params_ordered = []
                best_idx = int(study.best_trial.params[f"ch_seq_idx_L{L}"])
                best_chs = ch_sequences[best_idx]
                for i in range(L):
                    ks = int(study.best_trial.params[f"layer_{i}_ks"])
                    ch = int(best_chs[i])
                    best_params_ordered.append((ks, ch))

                if verbose:
                    print(f"[Gen] Best {study_name}: value={study.best_value:.6f} specs={best_params_ordered}")

                results[(L, S)] = {
                    "best_params": tuple(best_params_ordered),
                    "best_value": float(study.best_value),
                    "study": study,
                }

        return results

    @staticmethod
    def _is_input_divisible(S: int, L: int) -> bool:
        if L < 1:
            return False
        divisor = 2 ** (L - 1)
        return (S % divisor) == 0

    @staticmethod
    def perform_param_sanity_check(input_size: int, specs: Sequence[EncoderSpec]) -> Tuple[bool, str]:
        L = len(specs)
        if L == 0:
            return False, "no layers"

        if not UNetParamGenerator._is_input_divisible(input_size, L):
            return False, f"input_size {input_size} not divisible by 2**(L-1) with L={L}"

        prev_ch = -math.inf
        for i, (ks, ch) in enumerate(specs):
            if ks % 2 == 0 or ks < 3:
                return False, f"even or too-small kernel_size at layer {i}: {ks}"
            if ch <= prev_ch:
                return False, f"channels not strictly increasing at layer {i}: {ch} <= {prev_ch}"
            prev_ch = ch

        return True, "ok"

    def evaluate_model(self, input_size: int, specs: Sequence[EncoderSpec]) -> float:
        if self.evaluator is None:
            raise ValueError("evaluator is None")
        return float(self.evaluator.evaluate(input_size, list(specs)))

    def _get_channel_sequences(self, L: int) -> list[tuple[int, ...]]:
        chans = sorted(set(int(c) for c in self.num_layer_channels))
        return list(combinations(chans, L))
