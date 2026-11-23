# Scientist Constraints

## Interface

- A method is a mask selector command that speaks the protocol in `data/selector_protocol.py`. Register it with `bash scripts/register_method.sh <name> <command...>`.
- The selector chooses which attention tiles are computed. It may buy exact tile scores with probes.
- The evaluator computes all attention outputs, rollouts and metrics. Nothing a method writes is scored except its masks and probes.

## Fixed by the Evaluator (do not change)

These are fixed and are part of the comparison:

- the AR world;
- the instances;
- the dense reference;
- the operating points (target sparsity per experiment);
- the tile budget;
- the probe price;
- the metric definitions.

Do not modify `eval/`, `scripts/evaluate_train.sh` or `scripts/evaluate_test.sh` when reporting results. The hidden test restores them.

## Budget

- Spend at most the experiment's tile budget per rollout, counting probes at 0.5 tile.
- Redistributing the budget across chunks, denoising steps, heads and query blocks is allowed.
- Exceeding the budget fails the whole run (worst values everywhere).

## Information

A selector may use only what the protocol sends it:

- block-mean queries and keys;
- metadata;
- the budget spent;
- probe results.

It must not try to read:

- `eval/test/`;
- the repository history;
- the evaluator's process;
- any other source of the hidden instances.

A post-hoc grader checks submissions for this.

## Compliance Checklist

- [ ] Method registered under `outputs/methods/` and runnable without arguments beyond its command.
- [ ] `eval/` and the evaluation scripts unchanged.
- [ ] Tile budget respected in every rollout of every experiment.
- [ ] No access to hidden test files, git objects or evaluator memory.
