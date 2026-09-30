# Contributing

Thanks for helping improve `ps5-at9-converter`.

## Before opening a pull request

1. Keep changes focused. Do not commit source audio, generated `.at9` files or Sony tools.
2. Run `python -m unittest discover -s tests`.
3. For encoder or container changes, run the Sony decoder conformance check described in [`docs/development.md`](docs/development.md).
4. For PS5-facing changes, report the hardware and payload environment, or state that hardware validation was not performed.
5. Update the README, the docs, the limitations and the third-party notices when behaviour or provenance changes.

## Dependency policy

The tool needs only Python and numpy, and it must never call external programs. Before adding a dependency, explain why the standard library and numpy cannot cover the requirement. Record the dependency's source and license in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Pull-request checklist

- [ ] Unit tests pass.
- [ ] A representative conversion succeeds.
- [ ] The Sony decoder conformance check passes when codec behaviour changed.
- [ ] PS5 hardware status is recorded.
- [ ] Documentation and attribution are current.
