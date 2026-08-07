# target-recall-hnsw

> 🚧 Work in progress. Following a structured build handbook — see progress in commit history.

A from-scratch HNSW (Hierarchical Navigable Small World) vector index for
Python, where you specify a **target recall** instead of hand-tuning
`ef_search`.

Every standard HNSW implementation makes you pick one fixed `ef_search`
value and use it for every query - wasting latency on easy queries,
under-delivering accuracy on hard ones. This library adaptively decides,
per query, how hard to search: stopping early once additional effort stops
improving the result, using an online feedback controller instead of a
one-size-fits-all constant.

Inspired by recent research on declarative-recall ANN search (Ada-ef,
SIGMOD 2026) - this implementation uses a different, self-designed
stopping heuristic rather than reproducing that paper's method.

## Status

- [ ] Core graph (distance functions, level sampler, graph storage)
- [ ] `search_layer` and neighbor selection heuristic
- [ ] `HNSWIndex` (insert + fixed-ef_search baseline)
- [ ] `AdaptiveEfController`
- [ ] Benchmark suite: recall-vs-latency, fixed vs. adaptive
- [ ] Published results in `docs/BENCHMARKS.md`
- [ ] v1.0.0 on PyPI

## License

MIT